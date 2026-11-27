"""Stage V ASI Subsystem: Bayesian Log-Odds & PRM-Weighted Swarm Consensus"""
import math

class BayesianSwarmConsensus:
    @staticmethod
    def aggregate_specialist_beliefs(agent_probs, agent_prm_reliabilities):
        # Weighted log-odds pooling: L = sum_k w_k * log(p_k / (1 - p_k))
        log_odds = 0.0
        for p, rel in zip(agent_probs, agent_prm_reliabilities):
            p_c = min(0.9999, max(0.0001, p))
            weight = max(0.0, 2.0 * rel - 1.0) ** 2
            log_odds += weight * math.log(p_c / (1.0 - p_c))
        consensus_p = 1.0 / (1.0 + math.exp(-log_odds))
        return consensus_p

if __name__ == "__main__":
    # 3 high-PRM verified specialists vs 2 noisy low-PRM hallucinations
    probs = [0.94, 0.92, 0.95, 0.15, 0.20]
    prms  = [0.98, 0.97, 0.99, 0.52, 0.51]
    p_star = BayesianSwarmConsensus.aggregate_specialist_beliefs(probs, prms)
    naive_mean = sum(probs) / len(probs)
    assert p_star > 0.999 and p_star > naive_mean
    print(f"METRIC:{p_star / naive_mean:.2f}")
