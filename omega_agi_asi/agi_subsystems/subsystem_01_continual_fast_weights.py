"""Stage I AGI Subsystem: Surprise-Gated Continual Fast-Weight Plasticity"""
import math

class SurpriseGatedFastWeightMemory:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.S = [[0.0] * dim for _ in range(dim)]

    def observe_and_adapt(self, k, v, base_decay: float = 0.95):
        kn = math.sqrt(sum(x*x for x in k)) or 1.0
        k = [x / kn for x in k]
        pred = [base_decay * sum(self.S[i][j] * k[j] for j in range(self.dim)) for i in range(self.dim)]
        surprise = math.sqrt(sum((vi - pi)**2 for vi, pi in zip(v, pred)))
        # Surprise-modulated plasticity gate beta_t in (0.75, 1.0)
        beta = 1.0 / (1.0 + math.exp(-4.0 * (surprise + 0.3)))
        for i in range(self.dim):
            delta_i = beta * (v[i] - pred[i])
            for j in range(self.dim):
                self.S[i][j] = base_decay * self.S[i][j] + delta_i * k[j]
        recall = [sum(self.S[i][j] * k[j] for j in range(self.dim)) for i in range(self.dim)]
        post_err = math.sqrt(sum((vi - ri)**2 for vi, ri in zip(v, recall)))
        return {"surprise": surprise, "beta_gate": beta, "post_err": post_err}

if __name__ == "__main__":
    mem = SurpriseGatedFastWeightMemory(dim=8)
    k = [1.0, -0.5, 0.25, 0.8, -0.2, 0.1, 0.4, -0.7]
    v = [0.9, -0.4, 0.3, -0.8, 0.5, 0.2, -0.6, 0.7]
    r1 = mem.observe_and_adapt(k, v)
    r2 = mem.observe_and_adapt(k, v)
    assert r2["post_err"] < r1["surprise"] * 0.05
    print(f"METRIC:{r1['surprise'] / max(r2['post_err'], 1e-9):.2f}")
