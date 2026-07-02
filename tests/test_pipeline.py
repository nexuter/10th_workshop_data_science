import unittest
from tempfile import TemporaryDirectory

import numpy as np

from wildfire_llm_agent.ablation import without_physical_warm_start
from wildfire_llm_agent.agent import WildfireAgent
from wildfire_llm_agent.baselines import pso_only
from wildfire_llm_agent.cli import _random_sanity_predictions
from wildfire_llm_agent.metrics import average_precision_score, evaluate, recall_at_new_burn_area
from wildfire_llm_agent.pixel_calibration import FEATURE_NAMES, mask_feature_channels
from wildfire_llm_agent.pso_adapter import PsoWarmStartGenerator
from wildfire_llm_agent.renderer import ToolRenderer
from wildfire_llm_agent.schemas import CorrectionOperation, CorrectionPlan
from wildfire_llm_agent.synthetic import make_synthetic_case


class WildfirePipelineTest(unittest.TestCase):
    def test_agent_preserves_current_burn_and_shapes(self) -> None:
        inputs, _target = make_synthetic_case()
        output = WildfireAgent().predict(inputs)
        self.assertEqual(output.predicted_binary_burn_map_t_plus_h.shape, inputs.burn_map_t.shape)
        self.assertTrue(np.all(output.predicted_binary_burn_map_t_plus_h[inputs.burn_map_t > 0] == 1))
        self.assertGreaterEqual(output.correction_plan.confidence, 0.0)
        self.assertLessEqual(output.correction_plan.confidence, 1.0)

    def test_agent_runs_against_metrics(self) -> None:
        inputs, target = make_synthetic_case()
        output = WildfireAgent().predict(inputs)
        self.assertIn("wind_speed_and_direction", output.correction_plan.evidence_checklist)
        metrics = evaluate(output.predicted_binary_burn_map_t_plus_h, target, inputs.burn_map_t, inputs.weather_t.wind_direction_deg)
        for key in ("iou", "f1", "new_f1", "boundary_f1"):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)
        self.assertGreaterEqual(metrics["burned_area_error"], 0.0)

    def test_renderer_rejects_unknown_operation(self) -> None:
        inputs, _target = make_synthetic_case()
        plan = CorrectionPlan(
            operations=[CorrectionOperation("paint_arbitrary_pixels")],
            confidence=0.5,
            rationale="bad plan",
        )
        with self.assertRaises(ValueError):
            ToolRenderer().render(inputs, plan)

    def test_pso_only_is_monotonic(self) -> None:
        inputs, _target = make_synthetic_case()
        pred = pso_only(inputs)
        self.assertTrue(np.all(pred[inputs.burn_map_t > 0] == 1))

    def test_pso_event_id_parsing(self) -> None:
        self.assertEqual(PsoWarmStartGenerator.event_id_to_idx("0001_00012"), 12)
        self.assertEqual(PsoWarmStartGenerator.event_id_to_idx("35"), 35)

    def test_pso_effective_t_cap_uses_available_contiguous_frames(self) -> None:
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            scenario = Path(tmp) / "0002"
            mask_dir = scenario / "Satellite_Images_Mask" / "0002_00008"
            mask_dir.mkdir(parents=True)
            for index in range(1, 7):
                (mask_dir / f"out{index}.jpg").write_bytes(b"x")
            generator = PsoWarmStartGenerator(t_cap=48, cache_dir=None, include_user_site=False)
            self.assertEqual(generator._effective_t_cap(scenario, 8), 6)

    def test_llm_only_ablation_hides_physical_prior(self) -> None:
        inputs, _target = make_synthetic_case()
        llm_only = without_physical_warm_start(inputs)
        self.assertFalse(llm_only.metadata["uses_physical_warm_start"])
        self.assertEqual(llm_only.metadata["warm_start"], "none_current_burn_only")
        self.assertTrue(np.array_equal(llm_only.pso_forecast_t_plus_h, inputs.burn_map_t))
        self.assertTrue(np.array_equal(llm_only.burn_map_t, inputs.burn_map_t))

    def test_average_precision_rewards_ranked_probabilities(self) -> None:
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        good_scores = np.array([[0.9, 0.2], [0.8, 0.1]], dtype=float)
        bad_scores = np.array([[0.2, 0.9], [0.1, 0.8]], dtype=float)
        self.assertAlmostEqual(average_precision_score(good_scores, target), 1.0)
        self.assertLess(average_precision_score(bad_scores, target), 0.6)

    def test_recall_at_new_burn_area_rewards_top_ranked_spread(self) -> None:
        current = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        good_scores = np.array([[1.0, 0.9], [0.8, 0.1]], dtype=float)
        bad_scores = np.array([[1.0, 0.1], [0.2, 0.9]], dtype=float)
        self.assertAlmostEqual(recall_at_new_burn_area(good_scores, target, current), 1.0)
        self.assertLess(recall_at_new_burn_area(bad_scores, target, current), 1.0)

    def test_new_iou_does_not_reward_no_growth_frames(self) -> None:
        current = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        target = current.copy()
        prediction = current.copy()
        metrics = evaluate(prediction, target, current, wind_direction_deg=0.0)
        self.assertEqual(metrics["new_f1"], 0.0)
        self.assertEqual(metrics["new_iou"], 0.0)
        self.assertEqual(metrics["iou"], 1.0)

    def test_new_boundary_f1_ignores_existing_burn_boundary(self) -> None:
        current = np.zeros((5, 5), dtype=np.uint8)
        current[1:3, 1:3] = 1
        target = current.copy()
        target[2, 3] = 1
        prediction = current.copy()
        metrics = evaluate(prediction, target, current, wind_direction_deg=90.0)
        self.assertGreater(metrics["boundary_f1"], 0.0)
        self.assertEqual(metrics["new_boundary_f1"], 0.0)

    def test_feature_mask_only_removes_llm_channels(self) -> None:
        features = np.ones((2, 2, len(FEATURE_NAMES)), dtype=float)
        masked = mask_feature_channels(features, FEATURE_NAMES, ("agent_probability", "llm_only_probability"))
        self.assertTrue(np.all(masked[..., FEATURE_NAMES.index("agent_probability")] == 0.0))
        self.assertTrue(np.all(masked[..., FEATURE_NAMES.index("llm_only_probability")] == 0.0))
        self.assertTrue(np.all(masked[..., FEATURE_NAMES.index("warm_probability")] == 1.0))
        self.assertTrue(np.all(masked[..., FEATURE_NAMES.index("simple_probability")] == 1.0))

    def test_random_sanity_predictions_are_monotonic_and_budgeted(self) -> None:
        inputs, target = make_synthetic_case()
        predictions = _random_sanity_predictions(inputs=inputs, target=target, seed_parts=("unit", "case"))
        burn = inputs.burn_map_t > 0
        warm_new = int(((inputs.pso_forecast_t_plus_h > 0) & ~burn).sum())
        target_new = int(((target > 0) & ~burn).sum())
        for prediction in predictions.values():
            self.assertTrue(np.all(prediction[burn] == 1))
        self.assertEqual(int(((predictions["random_warm_area"] > 0) & ~burn).sum()), warm_new)
        self.assertEqual(int(((predictions["random_oracle_prevalence"] > 0) & ~burn).sum()), target_new)

    def test_docx_builder_resolves_latex_cross_references(self) -> None:
        try:
            from scripts.build_submission_docx import replace_cross_references, strip_tex
        except ModuleNotFoundError as exc:
            if exc.name == "docx":
                self.skipTest("python-docx is only available in the bundled document runtime")
            raise
        text = r"Figure~\ref{fig:success} and Table~\ref{tab:metrics} summarize this."
        resolved = replace_cross_references(text, {"fig:success": "3", "tab:metrics": "2"})
        self.assertEqual(strip_tex(resolved), "Figure 3 and Table 2 summarize this.")
        self.assertEqual(strip_tex(replace_cross_references(r"Figure~\ref{fig:missing}", {})), "Figure ??")


if __name__ == "__main__":
    unittest.main()
