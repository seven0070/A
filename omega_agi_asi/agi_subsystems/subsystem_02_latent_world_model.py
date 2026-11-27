"""Stage II AGI Subsystem: Counterfactual Latent World Model & Free-Energy Planner"""
import math

class CounterfactualLatentWorldModel:
    def __init__(self, dt: float = 0.05):
        self.dt = dt

    def rollout_policy(self, q0, p0, action_force, target_q, steps: int = 25):
        q, p = list(q0), list(p0)
        trajectory_cost = 0.0
        for _ in range(steps):
            # Symplectic kick-drift-kick under control force u
            p_h = [pi + 0.5 * self.dt * (ui - qi) for qi, pi, ui in zip(q, p, action_force)]
            q = [qi + self.dt * pi for qi, pi in zip(q, p_h)]
            p = [pi + 0.5 * self.dt * (ui - qi) for qi, pi, ui in zip(q, p_h, action_force)]
            dist2 = sum((qi - ti)**2 for qi, ti in zip(q, target_q))
            trajectory_cost += dist2
        return {"final_q": q, "cumulative_free_energy_cost": trajectory_cost / steps}

    def select_optimal_policy(self, q0, p0, target_q, candidate_actions):
        scored = [(act, self.rollout_policy(q0, p0, act, target_q)["cumulative_free_energy_cost"]) for act in candidate_actions]
        scored.sort(key=lambda x: x[1])
        return scored[0], scored[-1]

if __name__ == "__main__":
    wm = CounterfactualLatentWorldModel()
    best, worst = wm.select_optimal_policy(
        q0=[0.0, 0.0], p0=[0.0, 0.0], target_q=[1.0, -1.0],
        candidate_actions=[[0.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [0.5, -0.5]]
    )
    assert best[0] == [1.0, -1.0] and best[1] < worst[1]
    print(f"METRIC:{worst[1] / max(best[1], 1e-9):.2f}")
