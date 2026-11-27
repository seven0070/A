"""Comprehensive Verification Suite for OMEGA-GLM AGI & ASI Builder Engine."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_engine import (
    MathematicalPhysicsKernel,
    UniversalGLMBackbone,
    ProcessRewardVerifier,
    CognitiveTreeSearchEngine,
    OmegaAGIASICreatorEngine,
)


class TestOmegaAGIASIEngine(unittest.TestCase):
    def test_sinkhorn_knopp_birkhoff_doubly_stochastic(self):
        raw = [
            [2.5, 0.5, 0.1, 0.3],
            [0.2, 1.9, 0.4, 0.1],
            [0.3, 0.1, 3.1, 0.2],
            [0.1, 0.4, 0.2, 2.7],
        ]
        ds = MathematicalPhysicsKernel.sinkhorn_knopp_birkhoff(raw, iterations=30)
        for row in ds:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
        for j in range(4):
            col_sum = sum(ds[i][j] for i in range(4))
            self.assertAlmostEqual(col_sum, 1.0, places=5)

    def test_directional_abliteration_orthogonality_and_norm(self):
        w = [[3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 2.0, 0.0]]
        refusal_dir = [[1.0, 0.0, 0.0, 0.0]]
        w_ablit = MathematicalPhysicsKernel.directional_abliteration_projection(
            w, refusal_dir, strengths=[1.0], preserve_norm=True
        )
        # First component along [1,0,0,0] must be eliminated while L2 norm (5.0 and 3.0) is preserved
        self.assertAlmostEqual(w_ablit[0][0], 0.0, places=6)
        self.assertAlmostEqual(sum(x * x for x in w_ablit[0]) ** 0.5, 5.0, places=5)
        self.assertAlmostEqual(w_ablit[1][0], 0.0, places=6)
        self.assertAlmostEqual(sum(x * x for x in w_ablit[1]) ** 0.5, 3.0, places=5)

    def test_variational_free_energy_and_solomonoff(self):
        fe = MathematicalPhysicsKernel.variational_expected_free_energy(
            prior_probs=[0.25, 0.25, 0.25, 0.25],
            posterior_probs=[0.70, 0.10, 0.10, 0.10],
            utility_rewards=[1.0, 0.2, 0.0, -0.5],
        )
        self.assertGreater(fe["epistemic_info_gain_nats"], 0.0)
        self.assertLess(fe["expected_free_energy_G"], 0.0)

        sol = MathematicalPhysicsKernel.solomonoff_kolmogorov_prior("def f(x):\n    return x * x + 2 * x + 1\n")
        self.assertGreater(sol["kolmogorov_bits_est"], 0.0)

    def test_glm53_expert_routing_and_recursive_asi_loop(self):
        engine = OmegaAGIASICreatorEngine()
        self.assertEqual(engine.backbone.profile.num_hidden_layers, 45)
        self.assertEqual(engine.backbone.profile.routed_experts_pruned, 92)
        report = engine.recursive_self_improve(cycles=3)
        self.assertEqual(report["cycles_completed"], 3)
        self.assertGreater(report["final_asi_fitness"], 0.90)
        for item in report["evolution_history"]:
            self.assertTrue(item["accepted_into_core"])


if __name__ == "__main__":
    unittest.main()
