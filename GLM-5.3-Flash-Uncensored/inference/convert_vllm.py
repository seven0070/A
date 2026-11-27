"""Make the published checkpoint loadable by vLLM.
  1. every non-expert FP8 tensor (attention projections, shared experts, dense MLP) -> BF16.  vLLM's Glm5Next loader
     expects block-wise FP8 for attention and its FP8 linear kernel needs NVRTC; BF16 avoids both.
  2. routed experts: vLLM fuses gate_proj and up_proj of an expert into one w13 tensor and needs ONE global scale; the
     checkpoint has two different ones (ratio up to 2x).  Set both to min(Gg, Gu) and rescale the block scales of the
     projection that had the larger global scale (dequant = fp4 * S / G stays the same up to fp8 rounding of S).
usage: convert_vllm.py SRC DST"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
KEY = re.compile(r'^model\.language_model\.layers\.(\d+)\.(?:self_attn|mlp|mlp\.shared_experts)\.[a-z_]+\.weight$')
EXP = re.compile(r'^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+)\.(gate_proj|up_proj)\.(weight_scale|weight_global_scale)$')
idx = json.load(open(f'{SRC}/model.safetensors.index.json'))
wm, new_wm = idx['weight_map'], {}
handles = {}


def get(k):
    f = wm[k]
    if f not in handles:
        handles[f] = safe_open(f'{SRC}/{f}', 'pt', device='cpu')
    return handles[f].get_tensor(k)


fp8_dense = {k for k in wm if KEY.match(k) and (k[:-len('weight')] + 'weight_scale') in wm}
drop = {k[:-len('weight')] + 'weight_scale' for k in fp8_dense}
n_rescaled = 0

for shard in sorted(set(wm.values())):
    out = {}
    for k in [k for k, s in wm.items() if s == shard]:
        if k in drop:
            continue
        if k in fp8_dense:
            out[k] = (get(k).float() * get(k[:-len('weight')] + 'weight_scale').float()).to(torch.bfloat16)
            continue
        m = EXP.match(k)
        if m:
            base, proj, kind = m[1], m[2], m[3]
            gg = get(f'{base}.gate_proj.weight_global_scale').float().reshape(())
            gu = get(f'{base}.up_proj.weight_global_scale').float().reshape(())
            gm = torch.minimum(gg, gu)
            if kind == 'weight_global_scale':
                out[k] = gm.clone().reshape(get(k).shape)
            else:
                g = gg if proj == 'gate_proj' else gu
                ratio = gm / g
                if float(ratio) == 1.0:
                    out[k] = get(k)
                else:
                    out[k] = (get(k).float() * ratio).to(torch.float8_e4m3fn)
                    n_rescaled += 1
            continue
        out[k] = get(k)
    save_file(out, f'{DST}/{shard}', metadata={'format': 'pt'})
    for k in out:
        new_wm[k] = shard
    handles.clear()
    print('wrote', shard, len(out), flush=True)

idx['weight_map'] = new_wm
json.dump(idx, open(f'{DST}/model.safetensors.index.json', 'w'))
cfg = json.load(open(f'{SRC}/config.json'))
qc = cfg['quantization_config']
qc['config_groups'].pop('group_1', None)       # no FP8 left: only the NVFP4 routed experts remain quantized
qc['format'] = 'nvfp4-pack-quantized'
# the checkpoint has NO input (activation) scales: with input_activations set vLLM takes the W4A4 path and turns the
# missing scales into inf -> NaN logits.  Weight-only (W4A16, Marlin) is what this checkpoint really is.
qc['config_groups']['group_0']['input_activations'] = None
for k in sorted(fp8_dense):
    qc['ignore'].append(k.replace('model.language_model.', 'model.')[:-len('.weight')])
json.dump(cfg, open(f'{DST}/config.json', 'w'), indent=2)
for fn in os.listdir(SRC):
    if os.path.isfile(f'{SRC}/{fn}') and not fn.endswith('.safetensors') and fn not in ('config.json', 'model.safetensors.index.json'):
        shutil.copy(f'{SRC}/{fn}', f'{DST}/{fn}')
print('CONVERT DONE: dense fp8->bf16', len(fp8_dense), '| expert block-scale tensors rescaled', n_rescaled, flush=True)
