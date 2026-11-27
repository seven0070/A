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

    def test_real_npz_subspace_and_keep_json_surgery(self):
        from subspace_weight_surgery import RealSubspaceWeightSurgeryEngine
        surgery = RealSubspaceWeightSurgeryEngine()
        prof = surgery.inspect_subspace()
        self.assertEqual(prof.hidden_dim, 4096)
        self.assertEqual(prof.num_layers, 45)
        self.assertAlmostEqual(prof.q0_norm, 1.0, places=4)
        self.assertAlmostEqual(prof.q1_norm, 1.0, places=4)
        self.assertAlmostEqual(prof.q0_q1_dot_orthogonality, 0.0, places=4)
        stats = surgery.compute_expert_topology_statistics()
        self.assertEqual(stats["moe_layers_analyzed"], 42)
        self.assertEqual(stats["experts_kept_per_layer"], 92)

    def test_darwin_godel_ast_self_rewriter(self):
        from darwin_godel_rewriter import DarwinGodelSelfRewriter
        rewriter = DarwinGodelSelfRewriter()
        res = rewriter.run_self_rewriting_evolution()
        self.assertEqual(res["mutations_evaluated"], 3)
        self.assertEqual(res["mutations_committed"], 3)
        self.assertGreater(res["results"][0]["error_improvement_factor"], 35.0)

    def test_autonomous_curriculum_and_sqlite_memory_vault(self):
        from autonomous_curriculum_memory import AutonomousCurriculumEngine
        cur = AutonomousCurriculumEngine()
        res = cur.run_self_play_curriculum()
        self.assertEqual(res["curriculum_tiers_completed"], 4)
        self.assertGreaterEqual(res["total_verified_skills_in_vault"], 4)
        for item in res["curriculum_results"]:
            self.assertTrue(item["passed"])

    def test_agi_asi_architect_5_bridge_subsystems(self):
        from agi_asi_architect import AGIASIArchitectBuilder
        builder = AGIASIArchitectBuilder()
        report = builder.build_all_agi_asi_subsystems()
        self.assertTrue(report["all_subsystems_verified"])
        self.assertEqual(report["subsystems_count"], 5)

    def test_universal_agi_builder_8_cognitive_faculties(self):
        from agi_builder import UniversalAGIBuilder
        builder = UniversalAGIBuilder()
        bp = builder.build_agi(name="Atlas-GLM-AGI-v1")
        self.assertTrue(bp.all_faculties_verified)
        self.assertEqual(len(bp.faculties_enabled), 8)
        self.assertEqual(
            bp.cognitive_evaluation["faculty_5_analogical_program_induction"]["zero_shot_transferred_output"],
            [4, 3, 8, 10, 20],
        )

    def test_thermodynamic_asi_swarm_and_master_pipeline(self):
        from swarm_agent import ThermodynamicSwarmOrchestrator
        swarm = ThermodynamicSwarmOrchestrator()
        rep = swarm.execute_swarm_mission("Synthesize multi-agent consensus kernel")
        self.assertEqual(rep["swarm_size"], 5)
        self.assertEqual(rep["nodes_passed_sandbox"], 5)
        self.assertGreater(rep["swarm_bayesian_consensus_probability"], 0.985)

    def test_open_ended_ast_genetic_synthesizer_and_auto_repair(self):
        from open_ended_synthesizer import run_side_two_demonstration
        rep = run_side_two_demonstration()
        self.assertTrue(rep["all_challenges_solved"])
        self.assertEqual(len(rep["challenges"]), 3)


if __name__ == "__main__":
    unittest.main()
