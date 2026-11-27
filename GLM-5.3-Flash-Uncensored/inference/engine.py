"""Lossless single-GPU inference for GLM-5.3-Flash-Uncensored-NVFP4 (191 GiB) on a 96 GB GPU + big host RAM.

All 288 routed experts per layer are kept in NVFP4-packed form.  The first GPU_LAYERS layers keep them on the GPU,
the rest live in host RAM; at run time only the experts the router actually selects are gathered, copied to the GPU
and dequantised to bf16 on the fly.  Everything else (attention, shared experts, dense layers, router) is bf16 on GPU.
"""
import gc, json, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextDecoderLayer, Glm5NextTextModel)

DEV = 'cuda'
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=DEV)


_B = torch.arange(256, device=DEV)
LUT = torch.stack([FP4[_B & 15], FP4[_B >> 4]], 1).to(torch.bfloat16)      # byte -> two fp4 values (low nibble first)


def deq(P, S, G):
    """NVFP4 -> bf16.  P uint8 [e,o,i/2], S fp8-as-uint8 [e,o,i/16], G fp32 [e]"""
    e, o, i2 = P.shape
    w = torch.index_select(LUT, 0, P.reshape(-1).int()).view(e, o, i2 * 2 // 16, 16)
    sc = (S.view(torch.float8_e4m3fn).float() / G.float().view(e, 1, 1)).to(torch.bfloat16)
    return (w * sc.unsqueeze(-1)).view(e, o, i2 * 2)


class PackedExperts(torch.nn.Module):
    """Drop-in for Glm5NextTextExperts that keeps NVFP4-packed weights (on GPU or host) and dequantises on demand."""
    PROJ = ('gate_proj', 'up_proj', 'down_proj')

    def __init__(self, cfg, store, on_gpu):
        super().__init__()
        self.E, self.on_gpu, self.limit = cfg.n_routed_experts, on_gpu, cfg.swiglu_limit
        self.store = store          # {proj: (P, S, G)}
        self.CH = 64

    _stage = {}

    @classmethod
    def _to_gpu(cls, t, ids):
        """Gather rows `ids` of a host tensor through a pinned staging buffer and copy them to the GPU."""
        key = (tuple(t.shape[1:]), t.dtype)
        buf = cls._stage.get(key)
        if buf is None:
            try:
                buf = torch.empty((64,) + tuple(t.shape[1:]), dtype=t.dtype, pin_memory=True)
            except Exception:
                buf = torch.empty((64,) + tuple(t.shape[1:]), dtype=t.dtype)
            cls._stage[key] = buf
        out = buf[:len(ids)]
        torch.index_select(t, 0, ids, out=out)
        return out.to(DEV)

    def _fetch(self, ids):
        out = {}
        for proj, (P, S, G) in self.store.items():
            if self.on_gpu:
                i = ids.to(DEV)
                out[proj] = deq(P[i], S[i], G[i])
            else:
                out[proj] = deq(self._to_gpu(P, ids), self._to_gpu(S, ids), G[ids].to(DEV))
        return out

    def forward(self, hs, top_k_index, top_k_weights):
        if hs.shape[0] == 1:                       # single-token decode fast path
            W = self._fetch(top_k_index[0].cpu())
            x = hs[0]
            g = torch.einsum('koh,h->ko', W['gate_proj'], x).clamp(max=self.limit)
            u = torch.einsum('koh,h->ko', W['up_proj'], x).clamp(-self.limit, self.limit)
            y = torch.einsum('kho,ko->kh', W['down_proj'], F.silu(g) * u)
            return (y * top_k_weights[0][:, None].to(y.dtype)).sum(0, keepdim=True)
        k = top_k_index.shape[1]
        flat = top_k_index.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(inv, minlength=len(uniq)).tolist()
        tok = order // k
        w = top_k_weights.reshape(-1)[order]
        uc = uniq.cpu()
        out = torch.zeros_like(hs)
        s = 0
        for c0 in range(0, len(uc), self.CH):
            ids = uc[c0:c0 + self.CH]
            W = self._fetch(ids)
            for j in range(len(ids)):
                c = counts[c0 + j]
                sel = tok[s:s + c]; ww = w[s:s + c]; s += c
                g = F.linear(hs[sel], W['gate_proj'][j]).clamp(max=self.limit)
                u = F.linear(hs[sel], W['up_proj'][j]).clamp(-self.limit, self.limit)
                y = F.linear(F.silu(g) * u, W['down_proj'][j])
                out.index_add_(0, sel, (y * ww[:, None].to(y.dtype)).to(out.dtype))
            del W
        return out


def ck_key(k):
    if k.startswith('attn_hc.'): return 'hc_attn_' + k[8:]
    if k.startswith('ffn_hc.'): return 'hc_ffn_' + k[7:]
    if k.startswith('self_attn.forget_gate.'): return 'self_attn.' + k[len('self_attn.forget_gate.'):]
    return k


class Engine:
    def __init__(self, root='/tmp/base', gpu_layers=14, log=print):
        self.root, self.log = root, log
        cfg = AutoConfig.from_pretrained(root)
        self.tcfg = tc = cfg.text_config
        tc._attn_implementation = 'sdpa'
        self.tok = AutoTokenizer.from_pretrained(root)
        wm = json.load(open(f'{root}/model.safetensors.index.json'))['weight_map']
        self.wm, self._h = wm, {}
        t0 = time.time()
        with torch.device('meta'):
            self.model = Glm5NextTextModel(tc)
        m = self.model
        m.embed_tokens.weight = torch.nn.Parameter(self.get('model.language_model.embed_tokens.weight'), requires_grad=False)
        m.norm.weight = torch.nn.Parameter(self.get('model.language_model.norm.weight').float(), requires_grad=False)
        self.lm = self.get('lm_head.weight')
        n_gpu = 0
        for li in range(tc.num_hidden_layers):
            on_gpu = tc.mlp_layer_types[li] == 'sparse' and n_gpu < gpu_layers
            n_gpu += on_gpu
            m.layers[li] = self.load_layer(li, on_gpu)
            self._h.clear()          # drop mmap handles: sandbox memory accounting counts mapped file pages
            gc.collect()
            if li % 4 == 0 or li == tc.num_hidden_layers - 1:
                log(f'loaded layer {li} ({"gpu" if on_gpu else "host"} experts) {time.time() - t0:.0f}s '
                    f'gpu {torch.cuda.memory_allocated() / 2**30:.1f}G')
        m.eval()
        for n, t in list(m.named_parameters()) + list(m.named_buffers()):
            assert not t.is_meta, n

    def get(self, k, dev=DEV):
        f = self.wm[k]
        if f not in self._h:
            self._h[f] = safe_open(f'{self.root}/{f}', 'pt', device='cpu')
        return self._h[f].get_tensor(k).to(dev)

    def get_w(self, k):
        t = self.get(k)
        if t.dtype == torch.float8_e4m3fn and k + '_scale' in self.wm:
            t = (t.float() * self.get(k + '_scale').float()).to(torch.bfloat16)
        return t

    def load_experts(self, pre, on_gpu):
        E = self.tcfg.n_routed_experts
        store = {}
        for proj in PackedExperts.PROJ:
            ts = {part: [self._h_get(f'{pre}mlp.experts.{e}.{proj}.{part}') for e in range(E)]
                  for part in ('weight_packed', 'weight_scale', 'weight_global_scale')}
            P = torch.stack(ts['weight_packed'])
            S = torch.stack([t.view(torch.uint8) for t in ts['weight_scale']])
            G = torch.stack([t.reshape(()) for t in ts['weight_global_scale']]).float()
            if on_gpu:
                P, S, G = P.to(DEV), S.to(DEV), G.to(DEV)
            store[proj] = (P, S, G)
        return PackedExperts(self.tcfg, store, on_gpu)

    def _h_get(self, k):
        return self.get(k, 'cpu')

    def load_layer(self, i, on_gpu):
        tc = self.tcfg
        pre = f'model.language_model.layers.{i}.'
        with torch.device('meta'):
            layer = Glm5NextTextDecoderLayer(tc, i)
        sparse = tc.mlp_layer_types[i] == 'sparse'
        if sparse:
            layer.mlp.experts = self.load_experts(pre, on_gpu)
        sd = {}
        for k in layer.state_dict():
            if k == 'self_attn.conv1d.weight':
                sd[k] = torch.cat([self.get(pre + f'self_attn.{c}_conv1d.weight') for c in 'qkv'], 0)
            else:
                sd[k] = self.get_w(pre + ck_key(k))
        layer.load_state_dict(sd, strict=True, assign=True)
        gc.collect()
        return layer.eval()

    @torch.no_grad()
    def logits(self, ids, cache, chunk=2048):
        """Feed ids [1,T] (in chunks) and return logits for the last position."""
        for s in range(0, ids.shape[1], chunk):
            out = self.model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1].to(torch.bfloat16)
        return F.linear(h, self.lm).float()[0]

    @torch.no_grad()
    def generate(self, ids, max_new=256, temperature=0.0, top_p=0.95, stop=None, stream=None):
        from transformers import DynamicCache
        cache = DynamicCache(config=self.tcfg)
        ids = torch.tensor([ids], device=DEV)
        stop = set(stop or self.tcfg.eos_token_id if isinstance(self.tcfg.eos_token_id, list) else [self.tcfg.eos_token_id])
        new, t0 = [], time.time()
        for n in range(max_new):
            lg = self.logits(ids, cache)
            if temperature > 0:
                p = torch.softmax(lg / temperature, -1)
                sp, si = p.sort(descending=True)
                keep = sp.cumsum(0) - sp <= top_p
                sp = sp * keep
                nxt = si[torch.multinomial(sp / sp.sum(), 1)]
            else:
                nxt = lg.argmax(-1, keepdim=True)
            t = int(nxt)
            if t in stop:
                break
            new.append(t)
            if stream:
                stream(t)
            ids = nxt.view(1, 1)
            if n == 0:
                t_first = time.time() - t0
        dt = time.time() - t0
        return new, {'prefill_s': round(t_first, 1) if new else None, 'tok_s': round(len(new) / max(dt - (t_first if new else 0), 1e-6), 2)}

    def chat(self, messages, tools=None, **kw):
        p = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = self.tok(p, add_special_tokens=False)['input_ids']
        new, st = self.generate(ids, **kw)
        return self.tok.decode(new, skip_special_tokens=False), st, len(ids)
