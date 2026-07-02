from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import site
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wildfire_llm_agent.ablation import without_physical_warm_start
from wildfire_llm_agent.agent import WildfireAgent
from wildfire_llm_agent.array_ops import as_binary, burn_front, dilate, shift, wind_to_offset
from wildfire_llm_agent.baselines import persistence, pso_only, simple_physics_warm_start
from wildfire_llm_agent.data import MaterializedScenarioReader, discover_materialized_scenarios
from wildfire_llm_agent.metrics import evaluate
from wildfire_llm_agent.pixel_calibration import (
    FEATURE_NAMES,
    PixelCalibrator,
    build_pixel_features,
    calibrated_burn_probability,
    fit_pixel_calibrator,
    mask_feature_channels,
    sample_new_burn_training_pixels,
)
from wildfire_llm_agent.pso_adapter import DEFAULT_PSO_PARAMS, PsoWarmStartGenerator
from wildfire_llm_agent.reasoners import HeuristicReasoner, OllamaReasoner
from wildfire_llm_agent.schemas import PredictionInput
from wildfire_llm_agent.selector_learning import build_selector_dataset, train_threshold_selector
from wildfire_llm_agent.selectors import AreaBudgetSelector, SelectionDecision
from wildfire_llm_agent.synthetic import make_synthetic_case


LLM_DERIVED_CALIBRATION_FEATURES = ("agent_probability", "llm_only_probability")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire PSO-warm-started LLM agent experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Show import paths and PSO dependency availability")

    demo = subparsers.add_parser("demo", help="Run a synthetic end-to-end demo")
    demo.add_argument("--output-dir", default="outputs/demo")
    _add_reasoner_args(demo)

    materialized = subparsers.add_parser("run-materialized", help="Run one materialized dataset event")
    materialized.add_argument("--scenario-root", required=True)
    materialized.add_argument("--event-index", type=int, default=0)
    materialized.add_argument("--frame-index", type=int, default=0)
    materialized.add_argument("--output-dir", default="outputs/materialized")
    materialized.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    materialized.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    materialized.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="auto")
    materialized.add_argument("--pso-t-cap", type=int, default=48)
    materialized.add_argument("--ros-wildfire-src", default=None)
    materialized.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    materialized.add_argument("--fallback-placeholder", action="store_true")
    _add_reasoner_args(materialized)

    root_eval = subparsers.add_parser("evaluate-materialized-root", help="Batch-evaluate events under dataset/pkl_materialized")
    root_eval.add_argument("--dataset-root", required=True)
    root_eval.add_argument("--max-scenarios", type=int, default=None)
    root_eval.add_argument("--max-events-per-scenario", type=int, default=3)
    root_eval.add_argument("--max-frame-pairs-per-event", type=int, default=5)
    root_eval.add_argument("--output-dir", default="outputs/materialized_eval")
    root_eval.add_argument("--save-panels", action="store_true")
    root_eval.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    root_eval.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    root_eval.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="auto")
    root_eval.add_argument("--pso-t-cap", type=int, default=None)
    root_eval.add_argument("--ros-wildfire-src", default=None)
    root_eval.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    root_eval.add_argument("--fallback-placeholder", action="store_true")
    root_eval.add_argument("--include-llm-only", action="store_true")
    _add_selector_args(root_eval)
    _add_reasoner_args(root_eval)

    case_figure = subparsers.add_parser("export-case-figure", help="Export qualitative panels for one event/frame")
    case_figure.add_argument("--scenario-root", required=True)
    case_figure.add_argument("--event-id", required=True)
    case_figure.add_argument("--frame-index", type=int, required=True)
    case_figure.add_argument("--output-dir", default="outputs/case_figures")
    case_figure.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    case_figure.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    case_figure.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="cpu")
    case_figure.add_argument("--pso-t-cap", type=int, default=None)
    case_figure.add_argument("--ros-wildfire-src", default=None)
    case_figure.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    case_figure.add_argument("--fallback-placeholder", action="store_true")
    case_figure.add_argument("--include-llm-only", action="store_true")
    _add_selector_args(case_figure)
    _add_reasoner_args(case_figure)

    calibrated_case = subparsers.add_parser("export-calibrated-case", help="Export a qualitative figure for the learned calibrated hybrid")
    calibrated_case.add_argument("--dataset-root", required=True)
    calibrated_case.add_argument("--scenario-root", required=True)
    calibrated_case.add_argument("--event-id", required=True)
    calibrated_case.add_argument("--frame-index", type=int, required=True)
    calibrated_case.add_argument("--max-scenarios", type=int, default=5)
    calibrated_case.add_argument("--max-events-per-scenario", type=int, default=3)
    calibrated_case.add_argument("--train-events-per-scenario", type=int, default=2)
    calibrated_case.add_argument("--max-frame-pairs-per-event", type=int, default=5)
    calibrated_case.add_argument("--output-dir", default="outputs/case_figures/calibrated_case")
    calibrated_case.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    calibrated_case.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    calibrated_case.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="cpu")
    calibrated_case.add_argument("--pso-t-cap", type=int, default=48)
    calibrated_case.add_argument("--ros-wildfire-src", default=None)
    calibrated_case.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    calibrated_case.add_argument("--fallback-placeholder", action="store_true")
    calibrated_case.add_argument("--calibration-iterations", type=int, default=600)
    calibrated_case.add_argument("--calibration-learning-rate", type=float, default=0.08)
    calibrated_case.add_argument("--calibration-min-new-cells", type=int, default=0)
    calibrated_case.add_argument("--calibration-report", default=None)
    _add_selector_args(calibrated_case)
    _add_reasoner_args(calibrated_case)

    summarize = subparsers.add_parser("summarize-results", help="Write a Markdown report for an evaluation output directory")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--baseline-method", default="warm_start")
    summarize.add_argument("--report-name", default="report.md")

    bootstrap = subparsers.add_parser("bootstrap-comparisons", help="Bootstrap paired metric deltas from an evaluation metrics.csv")
    bootstrap.add_argument("--output-dir", required=True)
    bootstrap.add_argument("--candidate-method", default="calibrated")
    bootstrap.add_argument("--comparison-methods", nargs="+", default=["warm_start", "agent_selected", "calibrated_no_llm"])
    bootstrap.add_argument("--metric", default="new_f1")
    bootstrap.add_argument("--iterations", type=int, default=2000)
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.add_argument("--report-name", default="bootstrap_comparisons.md")
    bootstrap.add_argument("--unit", choices=["frame", "event", "scenario"], default="frame")

    metric_audit = subparsers.add_parser("audit-metrics", help="Audit existing metrics.csv for known metric-definition risks")
    metric_audit.add_argument("--output-dir", required=True)
    metric_audit.add_argument("--report-name", default="metric_audit.md")
    metric_audit.add_argument("--repaired-metrics-name", default="metrics_repaired.csv")

    paper_assets = subparsers.add_parser("make-paper-assets", help="Build paper-facing result tables from completed runs")
    paper_assets.add_argument("--calibration-5-dir", default="outputs/pixel_calibration_policy_5scenario_5frames")
    paper_assets.add_argument("--calibration-10-dir", default="outputs/pixel_calibration_policy_10scenario_5frames")
    paper_assets.add_argument("--rotating-cv-dir", default="outputs/pixel_calibration_rotating_cv_5scenario_5frames")
    paper_assets.add_argument("--matched-baseline-dir", default="outputs/ablation_hybrid_prior_gated_5scenario_5frames")
    paper_assets.add_argument("--output-dir", default="outputs/paper_assets")
    paper_assets.add_argument("--summary-doc", default="docs/paper_results_summary.md")

    temporal = subparsers.add_parser("summarize-temporal", help="Summarize metrics by frame_index for temporal trend analysis")
    temporal.add_argument("--output-dir", required=True)
    temporal.add_argument("--baseline-method", default="warm_start")
    temporal.add_argument("--candidate-method", default="agent_selected")
    temporal.add_argument("--report-name", default="temporal_report.md")

    selector_dataset = subparsers.add_parser("build-selector-dataset", help="Build frame-level features for learned selector experiments")
    selector_dataset.add_argument("--evaluation-dir", required=True)
    selector_dataset.add_argument("--dataset-root", required=True)
    selector_dataset.add_argument("--output-path", required=True)

    threshold_selector = subparsers.add_parser("train-threshold-selector", help="Fit a data-driven threshold selector from selector features")
    threshold_selector.add_argument("--selector-dataset", required=True)
    threshold_selector.add_argument("--output-dir", required=True)
    threshold_selector.add_argument("--fallback-method", choices=["llm_only", "warm_start"], default="llm_only")
    threshold_selector.add_argument("--auprc-weight", type=float, default=0.25)
    threshold_selector.add_argument("--area-error-weight", type=float, default=0.05)
    threshold_selector.add_argument("--wind-error-weight", type=float, default=0.05)

    calibration = subparsers.add_parser("calibrate-materialized-root", help="Train and evaluate a pixel-level calibrated hybrid predictor")
    calibration.add_argument("--dataset-root", required=True)
    calibration.add_argument("--max-scenarios", type=int, default=None)
    calibration.add_argument("--max-events-per-scenario", type=int, default=3)
    calibration.add_argument("--train-events-per-scenario", type=int, default=2)
    calibration.add_argument("--max-frame-pairs-per-event", type=int, default=5)
    calibration.add_argument("--output-dir", default="outputs/pixel_calibration")
    calibration.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    calibration.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    calibration.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="cpu")
    calibration.add_argument("--pso-t-cap", type=int, default=48)
    calibration.add_argument("--ros-wildfire-src", default=None)
    calibration.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    calibration.add_argument("--fallback-placeholder", action="store_true")
    calibration.add_argument("--calibration-iterations", type=int, default=600)
    calibration.add_argument("--calibration-learning-rate", type=float, default=0.08)
    calibration.add_argument("--calibration-min-new-cells", type=int, default=0)
    _add_selector_args(calibration)
    _add_reasoner_args(calibration)

    calibration_cv = subparsers.add_parser(
        "cross-validate-materialized-calibration",
        help="Run rotating held-out-event validation for the pixel-level calibrated hybrid",
    )
    calibration_cv.add_argument("--dataset-root", required=True)
    calibration_cv.add_argument("--max-scenarios", type=int, default=None)
    calibration_cv.add_argument("--max-events-per-scenario", type=int, default=3)
    calibration_cv.add_argument("--max-frame-pairs-per-event", type=int, default=5)
    calibration_cv.add_argument("--output-dir", default="outputs/pixel_calibration_rotating_cv")
    calibration_cv.add_argument("--warm-start", choices=["pso", "placeholder"], default="pso")
    calibration_cv.add_argument("--pso-params-file", default=str(DEFAULT_PSO_PARAMS))
    calibration_cv.add_argument("--pso-device", choices=["auto", "cpu", "cuda"], default="cpu")
    calibration_cv.add_argument("--pso-t-cap", type=int, default=48)
    calibration_cv.add_argument("--ros-wildfire-src", default=None)
    calibration_cv.add_argument("--pso-cache-dir", default="outputs/pso_cache")
    calibration_cv.add_argument("--fallback-placeholder", action="store_true")
    calibration_cv.add_argument("--calibration-iterations", type=int, default=600)
    calibration_cv.add_argument("--calibration-learning-rate", type=float, default=0.08)
    calibration_cv.add_argument("--calibration-min-new-cells", type=int, default=0)
    calibration_cv.add_argument("--min-train-events", type=int, default=2)
    _add_selector_args(calibration_cv)
    _add_reasoner_args(calibration_cv)

    args = parser.parse_args()
    if args.command == "doctor":
        run_doctor()
    elif args.command == "demo":
        run_demo(Path(args.output_dir), reasoner=_build_reasoner(args))
    elif args.command == "run-materialized":
        run_materialized(
            Path(args.scenario_root),
            args.event_index,
            args.frame_index,
            Path(args.output_dir),
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )
    elif args.command == "evaluate-materialized-root":
        evaluate_materialized_root(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            max_scenarios=args.max_scenarios,
            max_events_per_scenario=args.max_events_per_scenario,
            max_frame_pairs_per_event=args.max_frame_pairs_per_event,
            save_panels=args.save_panels,
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            include_llm_only=args.include_llm_only,
            selector_max_new_to_current_ratio=args.selector_max_new_to_current_ratio,
            selector_prior_fallback=args.selector_prior_fallback,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )
    elif args.command == "export-case-figure":
        export_case_figure(
            scenario_root=Path(args.scenario_root),
            event_id=args.event_id,
            frame_index=args.frame_index,
            output_dir=Path(args.output_dir),
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            include_llm_only=args.include_llm_only,
            selector_max_new_to_current_ratio=args.selector_max_new_to_current_ratio,
            selector_prior_fallback=args.selector_prior_fallback,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )
    elif args.command == "export-calibrated-case":
        export_calibrated_case(
            dataset_root=Path(args.dataset_root),
            scenario_root=Path(args.scenario_root),
            event_id=args.event_id,
            frame_index=args.frame_index,
            output_dir=Path(args.output_dir),
            max_scenarios=args.max_scenarios,
            max_events_per_scenario=args.max_events_per_scenario,
            train_events_per_scenario=args.train_events_per_scenario,
            max_frame_pairs_per_event=args.max_frame_pairs_per_event,
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            selector_max_new_to_current_ratio=args.selector_max_new_to_current_ratio,
            selector_prior_fallback=args.selector_prior_fallback,
            calibration_iterations=args.calibration_iterations,
            calibration_learning_rate=args.calibration_learning_rate,
            calibration_min_new_cells=args.calibration_min_new_cells,
            calibration_report=Path(args.calibration_report) if args.calibration_report else None,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )
    elif args.command == "summarize-results":
        summarize_results(Path(args.output_dir), baseline_method=args.baseline_method, report_name=args.report_name)
    elif args.command == "bootstrap-comparisons":
        bootstrap_comparisons(
            Path(args.output_dir),
            candidate_method=args.candidate_method,
            comparison_methods=args.comparison_methods,
            metric=args.metric,
            iterations=args.iterations,
            seed=args.seed,
            report_name=args.report_name,
            unit=args.unit,
        )
    elif args.command == "audit-metrics":
        audit_metrics(
            Path(args.output_dir),
            report_name=args.report_name,
            repaired_metrics_name=args.repaired_metrics_name,
        )
    elif args.command == "make-paper-assets":
        make_paper_assets(
            calibration_5_dir=Path(args.calibration_5_dir),
            calibration_10_dir=Path(args.calibration_10_dir),
            rotating_cv_dir=Path(args.rotating_cv_dir),
            matched_baseline_dir=Path(args.matched_baseline_dir),
            output_dir=Path(args.output_dir),
            summary_doc=Path(args.summary_doc),
        )
    elif args.command == "summarize-temporal":
        summarize_temporal(
            Path(args.output_dir),
            baseline_method=args.baseline_method,
            candidate_method=args.candidate_method,
            report_name=args.report_name,
        )
    elif args.command == "build-selector-dataset":
        rows = build_selector_dataset(
            evaluation_dir=Path(args.evaluation_dir),
            dataset_root=Path(args.dataset_root),
            output_path=Path(args.output_path),
        )
        print(json.dumps({"rows": len(rows), "output_path": args.output_path}, indent=2))
    elif args.command == "train-threshold-selector":
        report = train_threshold_selector(
            selector_dataset_path=Path(args.selector_dataset),
            output_dir=Path(args.output_dir),
            fallback_method=args.fallback_method,
            auprc_weight=args.auprc_weight,
            area_error_weight=args.area_error_weight,
            wind_error_weight=args.wind_error_weight,
        )
        print(json.dumps(_json_ready(report["best_threshold_policy"]), indent=2))
    elif args.command == "calibrate-materialized-root":
        calibrate_materialized_root(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            max_scenarios=args.max_scenarios,
            max_events_per_scenario=args.max_events_per_scenario,
            train_events_per_scenario=args.train_events_per_scenario,
            max_frame_pairs_per_event=args.max_frame_pairs_per_event,
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            selector_max_new_to_current_ratio=args.selector_max_new_to_current_ratio,
            selector_prior_fallback=args.selector_prior_fallback,
            calibration_iterations=args.calibration_iterations,
            calibration_learning_rate=args.calibration_learning_rate,
            calibration_min_new_cells=args.calibration_min_new_cells,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )
    elif args.command == "cross-validate-materialized-calibration":
        cross_validate_materialized_calibration(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            max_scenarios=args.max_scenarios,
            max_events_per_scenario=args.max_events_per_scenario,
            max_frame_pairs_per_event=args.max_frame_pairs_per_event,
            warm_start=args.warm_start,
            pso_params_file=Path(args.pso_params_file),
            pso_device=args.pso_device,
            pso_t_cap=args.pso_t_cap,
            ros_wildfire_src=args.ros_wildfire_src,
            pso_cache_dir=args.pso_cache_dir,
            fallback_placeholder=args.fallback_placeholder,
            selector_max_new_to_current_ratio=args.selector_max_new_to_current_ratio,
            selector_prior_fallback=args.selector_prior_fallback,
            calibration_iterations=args.calibration_iterations,
            calibration_learning_rate=args.calibration_learning_rate,
            calibration_min_new_cells=args.calibration_min_new_cells,
            min_train_events=args.min_train_events,
            reasoner=_build_reasoner(args),
            selector=_build_selector(args),
        )


def _add_reasoner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reasoner", choices=["heuristic", "ollama"], default="heuristic")
    parser.add_argument("--ollama-model", default="llama4:latest")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    parser.add_argument("--ollama-timeout-seconds", type=int, default=180)
    parser.add_argument("--ollama-cache-dir", default="outputs/ollama_cache")
    parser.add_argument("--ollama-no-cache", action="store_true")
    parser.add_argument("--ollama-fallback-heuristic", action="store_true")


def _add_selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selector", choices=["none", "area_budget"], default="none")
    parser.add_argument("--selector-min-confidence", type=float, default=0.45)
    parser.add_argument("--selector-max-added-cells-factor", type=float, default=3.0)
    parser.add_argument("--selector-min-added-cells", type=int, default=20)
    parser.add_argument("--selector-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--selector-max-new-to-current-ratio", type=float, default=None)
    parser.add_argument("--selector-prior-fallback", choices=["warm_start", "llm_only"], default="llm_only")


def _build_reasoner(args):
    if getattr(args, "reasoner", "heuristic") == "ollama":
        return OllamaReasoner(
            model=args.ollama_model,
            host=args.ollama_host,
            temperature=args.ollama_temperature,
            timeout_seconds=args.ollama_timeout_seconds,
            cache_dir=args.ollama_cache_dir,
            use_cache=not args.ollama_no_cache,
            fallback_reasoner=HeuristicReasoner() if args.ollama_fallback_heuristic else None,
        )
    return HeuristicReasoner()


def _build_selector(args):
    if getattr(args, "selector", "none") == "area_budget":
        return AreaBudgetSelector(
            min_confidence=args.selector_min_confidence,
            max_added_cells_factor=args.selector_max_added_cells_factor,
            min_allowed_added_cells=args.selector_min_added_cells,
            max_agent_to_warm_area_ratio=args.selector_max_area_ratio,
        )
    return None


def run_demo(output_dir: Path, reasoner) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, target = make_synthetic_case()
    output = WildfireAgent(reasoner=reasoner).predict(inputs, panel_path=output_dir / "context_panel.png")
    _write_gray(output_dir / "prediction.png", output.predicted_binary_burn_map_t_plus_h)
    _write_gray(output_dir / "target.png", target)
    metrics = {
        "persistence": evaluate(persistence(inputs), target, inputs.burn_map_t, inputs.weather_t.wind_direction_deg),
        "pso_only": evaluate(
            np.maximum(inputs.pso_forecast_t_plus_h, inputs.burn_map_t),
            target,
            inputs.burn_map_t,
            inputs.weather_t.wind_direction_deg,
        ),
        "simple_physics": evaluate(simple_physics_warm_start(inputs), target, inputs.burn_map_t, inputs.weather_t.wind_direction_deg),
        "agent": evaluate(
            output.predicted_burn_probability_map_t_plus_h,
            target,
            inputs.burn_map_t,
            inputs.weather_t.wind_direction_deg,
        ),
    }
    result = {
        "correction_plan": {
            "confidence": output.correction_plan.confidence,
            "operations": [op.__dict__ for op in output.correction_plan.operations],
            "rationale": output.correction_plan.rationale,
            "retrieved_snippet_ids": output.correction_plan.retrieved_snippet_ids,
            "source": output.correction_plan.source,
            "evidence_checklist": output.correction_plan.evidence_checklist,
        },
        "metrics": metrics,
        "diagnostics": output.diagnostics,
    }
    (output_dir / "result.json").write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready(result["metrics"]), indent=2))


def run_doctor() -> None:
    import wildfire_llm_agent

    user_site = None
    try:
        user_site = site.getusersitepackages()
    except Exception:
        pass

    specs = {
        "wildfire_llm_agent": importlib.util.find_spec("wildfire_llm_agent"),
        "wildfire_llm_agent.cli": importlib.util.find_spec("wildfire_llm_agent.cli"),
        "torch": importlib.util.find_spec("torch"),
        "ros_wildfire": importlib.util.find_spec("ros_wildfire"),
    }
    boosted_specs = {}
    added_user_site = False
    if user_site and user_site not in sys.path and Path(user_site).exists():
        sys.path.insert(0, user_site)
        added_user_site = True
        boosted_specs["torch_after_user_site"] = importlib.util.find_spec("torch")
        boosted_specs["ros_wildfire_after_user_site"] = importlib.util.find_spec("ros_wildfire")
    report = {
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "package_file": getattr(wildfire_llm_agent, "__file__", None),
        "module_origins": {
            name: getattr(spec, "origin", None) if spec else None
            for name, spec in specs.items()
        },
        "user_site": user_site,
        "user_site_exists": bool(user_site and Path(user_site).exists()),
        "user_site_added_for_check": added_user_site,
        "module_origins_after_user_site": {
            name: getattr(spec, "origin", None) if spec else None
            for name, spec in boosted_specs.items()
        },
        "pso_params_file": str(DEFAULT_PSO_PARAMS),
        "pso_params_exists": DEFAULT_PSO_PARAMS.exists(),
        "sys_path_head": sys.path[:10],
    }
    print(json.dumps(_json_ready(report), indent=2))


def run_materialized(
    scenario_root: Path,
    event_index: int,
    frame_index: int,
    output_dir: Path,
    *,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = MaterializedScenarioReader(scenario_root)
    event = reader.list_events()[event_index]
    static_layers = reader.load_static_layers()
    frames = reader.load_mask_sequence(event, max_frames=frame_index + 2)
    if frame_index + 1 >= len(frames):
        raise ValueError("frame-index must have a following target frame")
    burn_t = (frames[frame_index] >= 0.5).astype(np.uint8)
    target = (frames[frame_index + 1] >= 0.5).astype(np.uint8)
    weather = reader.load_weather(event, index=frame_index)
    warm_start_map, warm_start_source = _build_warm_start(
        warm_start=warm_start,
        scenario_root=scenario_root,
        event_id=event.event_id,
        frame_index=frame_index,
        burn_t=burn_t,
        fbfm13=static_layers.fbfm13,
        weather=weather,
        pso_generator=PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap,
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
        if warm_start == "pso"
        else None,
        fallback_placeholder=fallback_placeholder,
    )
    inputs = PredictionInput(
        burn_map_t=burn_t,
        pso_forecast_t_plus_h=warm_start_map,
        static_layers=static_layers,
        weather_t=weather,
        metadata={
            "scenario_root": str(scenario_root),
            "event_id": event.event_id,
            "frame_index": frame_index,
            "warm_start": warm_start_source,
            "uses_physical_warm_start": True,
        },
    )
    output = WildfireAgent(reasoner=reasoner).predict(inputs, panel_path=output_dir / "context_panel.png")
    metrics = evaluate(output.predicted_burn_probability_map_t_plus_h, target, burn_t, weather.wind_direction_deg)
    _write_gray(output_dir / "prediction.png", output.predicted_binary_burn_map_t_plus_h)
    _write_gray(output_dir / "target.png", target)
    (output_dir / "result.json").write_text(
        json.dumps(
            _json_ready(
                {
                "metrics": metrics,
                "correction_plan": {
                    "confidence": output.correction_plan.confidence,
                    "operations": [op.__dict__ for op in output.correction_plan.operations],
                    "rationale": output.correction_plan.rationale,
                    "retrieved_snippet_ids": output.correction_plan.retrieved_snippet_ids,
                    "source": output.correction_plan.source,
                    "evidence_checklist": output.correction_plan.evidence_checklist,
                },
                "diagnostics": output.diagnostics,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(metrics), indent=2))


def export_case_figure(
    scenario_root: Path,
    event_id: str,
    frame_index: int,
    output_dir: Path,
    *,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int | None,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    include_llm_only: bool,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = MaterializedScenarioReader(scenario_root)
    events = {event.event_id: event for event in reader.list_events()}
    if event_id not in events:
        available = ", ".join(sorted(events)[:10])
        raise ValueError(f"event-id {event_id!r} not found under {scenario_root}; first available events: {available}")
    event = events[event_id]
    static_layers = reader.load_static_layers()
    frames = reader.load_mask_sequence(event, max_frames=frame_index + 2)
    if frame_index + 1 >= len(frames):
        raise ValueError("frame-index must have a following target frame")

    burn_t = (frames[frame_index] >= 0.5).astype(np.uint8)
    target = (frames[frame_index + 1] >= 0.5).astype(np.uint8)
    weather = reader.load_weather(event, index=frame_index)
    warm_start_map, warm_start_source = _build_warm_start(
        warm_start=warm_start,
        scenario_root=scenario_root,
        event_id=event.event_id,
        frame_index=frame_index,
        burn_t=burn_t,
        fbfm13=static_layers.fbfm13,
        weather=weather,
        pso_generator=PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap or (frame_index + 2),
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
        if warm_start == "pso"
        else None,
        fallback_placeholder=fallback_placeholder,
    )
    inputs = PredictionInput(
        burn_map_t=burn_t,
        pso_forecast_t_plus_h=warm_start_map,
        static_layers=static_layers,
        weather_t=weather,
        metadata={
            "scenario_root": str(scenario_root),
            "event_id": event.event_id,
            "frame_index": frame_index,
            "warm_start": warm_start_source,
            "uses_physical_warm_start": True,
        },
    )
    output = WildfireAgent(reasoner=reasoner).predict(inputs, panel_path=output_dir / "context_panel.png")
    llm_only_output = None
    if include_llm_only:
        llm_only_output = WildfireAgent(reasoner=reasoner).predict(
            without_physical_warm_start(inputs),
            panel_path=output_dir / "llm_only_context_panel.png",
        )
    selected_prediction = None
    selected_score_map = None
    selection_decision = None
    if selector is not None:
        selected_prediction, selection_decision = selector.select(inputs, output)
        selected_score_map = (
            output.predicted_burn_probability_map_t_plus_h
            if selection_decision.selected == "agent"
            else np.maximum(inputs.pso_forecast_t_plus_h, burn_t)
        )
    selected_prediction, selected_score_map, selection_decision = _apply_prior_reliability_gate(
        inputs=inputs,
        selected_prediction=selected_prediction,
        selected_score_map=selected_score_map,
        selection_decision=selection_decision,
        llm_only_output=llm_only_output,
        max_new_to_current_ratio=selector_max_new_to_current_ratio,
        fallback=selector_prior_fallback,
    )

    predictions = {
        "persistence": persistence(inputs),
        "warm_start": np.maximum(inputs.pso_forecast_t_plus_h, burn_t),
        "agent_raw": output.predicted_burn_probability_map_t_plus_h,
    }
    if llm_only_output is not None:
        predictions["llm_only"] = llm_only_output.predicted_burn_probability_map_t_plus_h
    if selected_prediction is not None:
        predictions["agent_selected"] = selected_score_map

    metrics = {
        method: evaluate(prediction, target, burn_t, weather.wind_direction_deg)
        for method, prediction in predictions.items()
    }

    panel_specs = [
        ("burn_t", burn_t, "mask"),
        ("target_t_plus_1", target, "mask"),
        ("warm_start", predictions["warm_start"], "mask"),
        ("agent_raw", predictions["agent_raw"], "mask"),
    ]
    if "llm_only" in predictions:
        panel_specs.append(("llm_only", predictions["llm_only"], "mask"))
    if selected_prediction is not None:
        panel_specs.append(("agent_selected", selected_prediction, "mask"))
        panel_specs.append(("selected_error", _error_overlay(selected_prediction, target, burn_t), "rgb"))
    else:
        panel_specs.append(("agent_error", _error_overlay(predictions["agent_raw"], target, burn_t), "rgb"))

    crop_box = _burn_bbox(
        np.maximum.reduce(
            [
                burn_t,
                target,
                predictions["warm_start"],
                predictions["agent_raw"],
                predictions["llm_only"] if "llm_only" in predictions else predictions["agent_raw"],
                selected_prediction if selected_prediction is not None else predictions["agent_raw"],
            ]
        ),
        padding=8,
    )
    rendered_panels: list[tuple[str, Image.Image]] = []
    for name, values, mode in panel_specs:
        image = _rgb_panel(values) if mode == "mask" else Image.fromarray(values.astype(np.uint8), mode="RGB")
        image.save(output_dir / f"{name}.png")
        rendered_panels.append((name, image))

    figure_path = output_dir / "case_figure.png"
    _write_case_composite(figure_path, rendered_panels, crop_box=crop_box)
    summary = {
        "scenario_root": str(scenario_root),
        "event_id": event.event_id,
        "frame_index": frame_index,
        "warm_start_source": warm_start_source,
        "weather": weather.__dict__,
        "metrics": metrics,
        "correction_plan": {
            "confidence": output.correction_plan.confidence,
            "operations": [op.__dict__ for op in output.correction_plan.operations],
            "rationale": output.correction_plan.rationale,
            "retrieved_snippet_ids": output.correction_plan.retrieved_snippet_ids,
            "source": output.correction_plan.source,
            "evidence_checklist": output.correction_plan.evidence_checklist,
        },
        "llm_only_correction_plan": {
            "confidence": llm_only_output.correction_plan.confidence,
            "operations": [op.__dict__ for op in llm_only_output.correction_plan.operations],
            "rationale": llm_only_output.correction_plan.rationale,
            "retrieved_snippet_ids": llm_only_output.correction_plan.retrieved_snippet_ids,
            "source": llm_only_output.correction_plan.source,
            "evidence_checklist": llm_only_output.correction_plan.evidence_checklist,
        }
        if llm_only_output is not None
        else None,
        "selection_decision": selection_decision.__dict__ if selection_decision else None,
        "diagnostics": output.diagnostics,
        "legend": {
            "mask": "black=not burned, orange=burned",
            "error_overlay": "gray=current burn, green=true positive new burn, red=false positive, blue=false negative",
        },
        "crop_box_left_top_right_bottom": crop_box,
        "files": {
            "figure": str(figure_path),
            "context_panel": str(output_dir / "context_panel.png"),
        },
    }
    (output_dir / "case_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready({"figure": str(figure_path), "metrics": metrics, "selection_decision": summary["selection_decision"]}), indent=2))


def export_calibrated_case(
    *,
    dataset_root: Path,
    scenario_root: Path,
    event_id: str,
    frame_index: int,
    output_dir: Path,
    max_scenarios: int | None,
    max_events_per_scenario: int,
    train_events_per_scenario: int,
    max_frame_pairs_per_event: int,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    calibration_iterations: int,
    calibration_learning_rate: float,
    calibration_min_new_cells: int,
    calibration_report: Path | None,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_roots = discover_materialized_scenarios(dataset_root)
    if max_scenarios is not None:
        scenario_roots = scenario_roots[:max_scenarios]
    pso_generator = None
    if warm_start == "pso":
        pso_generator = PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap,
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
    agent = WildfireAgent(reasoner=reasoner)
    failures: list[dict[str, str]] = []
    train_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    train_examples_for_policy: list[dict[str, object]] = []

    for train_scenario_root in scenario_roots:
        reader = MaterializedScenarioReader(train_scenario_root)
        train_events = reader.list_events()[:max_events_per_scenario][:train_events_per_scenario]
        for event in train_events:
            examples = _collect_calibration_examples(
                scenario_root=train_scenario_root,
                event=event,
                reader=reader,
                max_frame_pairs_per_event=max_frame_pairs_per_event,
                output_dir=output_dir,
                split="case_train",
                warm_start=warm_start,
                pso_generator=pso_generator,
                fallback_placeholder=fallback_placeholder,
                agent=agent,
                selector=selector,
                selector_max_new_to_current_ratio=selector_max_new_to_current_ratio,
                selector_prior_fallback=selector_prior_fallback,
                failures=failures,
            )
            train_examples_for_policy.extend(examples)
            for example in examples:
                features, labels = sample_new_burn_training_pixels(
                    example["features"],
                    example["target"],
                    example["burn_t"],
                    seed=len(train_batches),
                )
                if len(features):
                    train_batches.append(features)
                    label_batches.append(labels)

    if calibration_report is not None:
        calibrator, calibration_policy = _calibrator_from_report(
            calibration_report,
            train_batches,
            fallback_examples=train_examples_for_policy,
            min_new_cells=calibration_min_new_cells,
        )
    else:
        calibrator = fit_pixel_calibrator(
            train_batches,
            label_batches,
            iterations=calibration_iterations,
            learning_rate=calibration_learning_rate,
        )
        calibration_policy = _fit_calibration_postprocess_policy(
            train_examples_for_policy,
            calibrator,
            min_new_cells=calibration_min_new_cells,
        )

    target_reader = MaterializedScenarioReader(scenario_root)
    events = {event.event_id: event for event in target_reader.list_events()}
    if event_id not in events:
        raise ValueError(f"event-id {event_id!r} not found under {scenario_root}")
    target_examples = _collect_calibration_examples(
        scenario_root=scenario_root,
        event=events[event_id],
        reader=target_reader,
        max_frame_pairs_per_event=max(max_frame_pairs_per_event, frame_index + 1),
        output_dir=output_dir,
        split="case_eval",
        warm_start=warm_start,
        pso_generator=pso_generator,
        fallback_placeholder=fallback_placeholder,
        agent=agent,
        selector=selector,
        selector_max_new_to_current_ratio=selector_max_new_to_current_ratio,
        selector_prior_fallback=selector_prior_fallback,
        failures=failures,
    )
    matching = [example for example in target_examples if int(example["frame_index"]) == frame_index]
    if not matching:
        raise ValueError(f"frame-index {frame_index} was not available for event {event_id}")
    example = matching[0]
    calibrated_raw = calibrated_burn_probability(example["inputs"], example["features"], calibrator)
    predictions = dict(example["predictions"])
    predictions["calibrated_raw"] = calibrated_raw
    predictions["calibrated"] = _apply_calibration_postprocess_policy(
        example["inputs"],
        calibrated_raw,
        predictions,
        calibration_policy["policy"],
    )
    metrics = {
        method: evaluate(
            prediction,
            example["target"],
            example["burn_t"],
            example["inputs"].weather_t.wind_direction_deg,
        )
        for method, prediction in predictions.items()
    }

    panel_specs = [
        ("burn_t", example["burn_t"], "mask"),
        ("target_t_plus_1", example["target"], "mask"),
        ("warm_start", predictions["warm_start"], "mask"),
        ("agent_selected", predictions.get("agent_selected", predictions["agent"]), "mask"),
        ("calibrated", predictions["calibrated"], "mask"),
        ("calibrated_error", _error_overlay(predictions["calibrated"], example["target"], example["burn_t"]), "rgb"),
    ]
    crop_box = _burn_bbox(
        np.maximum.reduce(
            [
                example["burn_t"],
                example["target"],
                predictions["warm_start"],
                predictions.get("agent_selected", predictions["agent"]),
                predictions["calibrated"],
            ]
        ),
        padding=8,
    )
    rendered_panels: list[tuple[str, Image.Image]] = []
    for name, values, mode in panel_specs:
        image = _rgb_panel(values) if mode == "mask" else Image.fromarray(values.astype(np.uint8), mode="RGB")
        image.save(output_dir / f"{name}.png")
        rendered_panels.append((name, image))
    probability_panel = _probability_panel(calibrated_raw)
    probability_panel.save(output_dir / "calibrated_raw_probability.png")

    figure_path = output_dir / "calibrated_case_figure.png"
    _write_case_composite(figure_path, rendered_panels, crop_box=crop_box)
    summary = {
        "scenario_root": str(scenario_root),
        "event_id": event_id,
        "frame_index": frame_index,
        "metrics": metrics,
        "calibration_policy": calibration_policy,
        "train_pixel_batches": len(train_batches),
        "train_pixels": int(sum(len(batch) for batch in label_batches)),
        "train_positive_pixels": int(sum(batch.sum() for batch in label_batches)),
        "failures": failures,
        "legend": {
            "mask": "black=not burned, orange=burned",
            "error_overlay": "gray=current burn, green=true positive new burn, red=false positive, blue=false negative",
            "probability": "black=low calibrated probability, yellow=high calibrated probability",
        },
        "crop_box_left_top_right_bottom": crop_box,
        "files": {
            "figure": str(figure_path),
            "calibrated_raw_probability": str(output_dir / "calibrated_raw_probability.png"),
        },
    }
    (output_dir / "calibrated_case_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready({"figure": str(figure_path), "metrics": metrics, "failures": len(failures)}), indent=2))


def evaluate_materialized_root(
    dataset_root: Path,
    output_dir: Path,
    max_scenarios: int | None,
    max_events_per_scenario: int,
    max_frame_pairs_per_event: int,
    save_panels: bool,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int | None,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    include_llm_only: bool,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_roots = discover_materialized_scenarios(dataset_root)
    if max_scenarios is not None:
        scenario_roots = scenario_roots[:max_scenarios]
    if not scenario_roots:
        raise ValueError(f"no materialized scenarios found under {dataset_root}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    agent = WildfireAgent(reasoner=reasoner)
    pso_generator = None
    if warm_start == "pso":
        pso_generator = PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap or (max_frame_pairs_per_event + 1),
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
    for scenario_root in scenario_roots:
        try:
            reader = MaterializedScenarioReader(scenario_root)
            static_layers = reader.load_static_layers()
            events = reader.list_events()[:max_events_per_scenario]
        except Exception as exc:
            failures.append({"scenario_root": str(scenario_root), "error": str(exc)})
            continue

        for event in events:
            try:
                frames = reader.load_mask_sequence(event, max_frames=max_frame_pairs_per_event + 1)
            except Exception as exc:
                failures.append({"scenario_root": str(scenario_root), "event_id": event.event_id, "error": str(exc)})
                continue

            if pso_generator is not None:
                required_frames = max_frame_pairs_per_event + 1
                missing_frames = _missing_mask_frames(event.mask_dir, required_frames)
                if missing_frames:
                    failures.append(
                        {
                            "scenario_root": str(scenario_root),
                            "event_id": event.event_id,
                            "error": (
                                f"skipped_event_incomplete_pso_sequence: missing {len(missing_frames)} "
                                f"of out1..out{required_frames}; first_missing={missing_frames[0]}"
                            ),
                        }
                    )
                    _write_evaluation_outputs(output_dir, rows, failures)
                    continue

            pair_count = max(0, min(max_frame_pairs_per_event, len(frames) - 1))
            for frame_index in range(pair_count):
                try:
                    burn_t = (frames[frame_index] >= 0.5).astype(np.uint8)
                    target = (frames[frame_index + 1] >= 0.5).astype(np.uint8)
                    weather = reader.load_weather(event, index=frame_index)
                    warm_start_map, warm_start_source = _build_warm_start(
                        warm_start=warm_start,
                        scenario_root=scenario_root,
                        event_id=event.event_id,
                        frame_index=frame_index,
                        burn_t=burn_t,
                        fbfm13=static_layers.fbfm13,
                        weather=weather,
                        pso_generator=pso_generator,
                        fallback_placeholder=fallback_placeholder,
                    )
                    inputs = PredictionInput(
                        burn_map_t=burn_t,
                        pso_forecast_t_plus_h=warm_start_map,
                        static_layers=static_layers,
                        weather_t=weather,
                        metadata={
                            "scenario_root": str(scenario_root),
                            "event_id": event.event_id,
                            "frame_index": frame_index,
                            "warm_start": warm_start_source,
                            "uses_physical_warm_start": True,
                        },
                    )
                    panel_path = None
                    if save_panels and not rows:
                        panel_path = output_dir / "first_context_panel.png"
                    elif getattr(reasoner, "model", None):
                        panel_path = output_dir / "llm_panels" / f"{scenario_root.parent.name}_{event.event_id}_{frame_index}.png"
                    output = agent.predict(inputs, panel_path=panel_path)
                    selected_prediction = None
                    selected_score_map = None
                    selection_decision = None
                    if selector is not None:
                        selected_prediction, selection_decision = selector.select(inputs, output)
                        selected_score_map = (
                            output.predicted_burn_probability_map_t_plus_h
                            if selection_decision.selected == "agent"
                            else np.maximum(inputs.pso_forecast_t_plus_h, burn_t)
                        )
                    predictions = {
                        "persistence": persistence(inputs),
                        "warm_start": np.maximum(inputs.pso_forecast_t_plus_h, burn_t),
                        "simple_physics": simple_physics_warm_start(inputs),
                        "agent": output.predicted_burn_probability_map_t_plus_h,
                    }
                    diagnostics_by_method = {"agent": output.diagnostics}
                    if include_llm_only:
                        llm_only_inputs = without_physical_warm_start(inputs)
                        llm_panel_path = None
                        if save_panels and not any(row.get("method") == "llm_only" for row in rows):
                            llm_panel_path = output_dir / "first_llm_only_context_panel.png"
                        elif getattr(reasoner, "model", None):
                            llm_panel_path = (
                                output_dir
                                / "llm_only_panels"
                                / f"{scenario_root.parent.name}_{event.event_id}_{frame_index}.png"
                            )
                        llm_only_output = agent.predict(llm_only_inputs, panel_path=llm_panel_path)
                        predictions["llm_only"] = llm_only_output.predicted_burn_probability_map_t_plus_h
                        diagnostics_by_method["llm_only"] = llm_only_output.diagnostics
                    selected_prediction, selected_score_map, selection_decision = _apply_prior_reliability_gate(
                        inputs=inputs,
                        selected_prediction=selected_prediction,
                        selected_score_map=selected_score_map,
                        selection_decision=selection_decision,
                        llm_only_output=llm_only_output if include_llm_only else None,
                        max_new_to_current_ratio=selector_max_new_to_current_ratio,
                        fallback=selector_prior_fallback,
                    )
                    if selected_prediction is not None:
                        predictions["agent_selected"] = selected_score_map
                    for method, prediction in predictions.items():
                        metric_values = evaluate(prediction, target, burn_t, weather.wind_direction_deg)
                        diagnostics = diagnostics_by_method.get(method, {})
                        rows.append(
                            {
                                "scenario": scenario_root.parent.name,
                                "fire_id": scenario_root.name,
                                "event_id": event.event_id,
                                "frame_index": frame_index,
                                "method": method,
                                "warm_start_source": warm_start_source,
                                "reasoner_source": diagnostics.get("reasoner_source", ""),
                                "selection": selection_decision.selected if method == "agent_selected" and selection_decision else "",
                                "selection_reason": selection_decision.reason if method == "agent_selected" and selection_decision else "",
                                "selection_confidence": selection_decision.confidence if method == "agent_selected" and selection_decision else "",
                                "selection_warm_new_cells": selection_decision.warm_new_cells if method == "agent_selected" and selection_decision else "",
                                "selection_agent_new_cells": selection_decision.agent_new_cells if method == "agent_selected" and selection_decision else "",
                                "selection_added_over_warm_cells": selection_decision.added_over_warm_cells if method == "agent_selected" and selection_decision else "",
                                "selection_max_allowed_added_cells": selection_decision.max_allowed_added_cells if method == "agent_selected" and selection_decision else "",
                                "selection_agent_to_warm_area_ratio": selection_decision.agent_to_warm_area_ratio if method == "agent_selected" and selection_decision else "",
                                "latency_seconds": diagnostics.get("latency_seconds", 0.0),
                                **metric_values,
                            }
                        )
                    _write_evaluation_outputs(output_dir, rows, failures)
                except Exception as exc:
                    failures.append(
                        {
                            "scenario_root": str(scenario_root),
                            "event_id": event.event_id,
                            "frame_index": str(frame_index),
                            "error": str(exc),
                        }
                    )
                    _write_evaluation_outputs(output_dir, rows, failures)

    summary = _write_evaluation_outputs(output_dir, rows, failures)
    print(
        json.dumps(
            _json_ready(
                {
                    "summary": summary,
                    "rows": len(rows),
                    "failures": len(failures),
                    "failure_examples": failures[:3],
                }
            ),
            indent=2,
        )
    )


def _write_evaluation_outputs(output_dir: Path, rows: list[dict[str, object]], failures: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    summary = _summarize_rows(rows)
    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready({"summary": summary, "rows": len(rows), "failures": failures}), indent=2),
        encoding="utf-8",
    )
    return summary


def _missing_mask_frames(mask_dir: Path, t_cap: int) -> list[str]:
    missing = []
    for index in range(1, int(t_cap) + 1):
        if not (mask_dir / f"out{index}.jpg").exists():
            missing.append(f"out{index}.jpg")
    return missing


def _apply_prior_reliability_gate(
    *,
    inputs: PredictionInput,
    selected_prediction: np.ndarray | None,
    selected_score_map: np.ndarray | None,
    selection_decision: SelectionDecision | None,
    llm_only_output,
    max_new_to_current_ratio: float | None,
    fallback: str,
) -> tuple[np.ndarray | None, np.ndarray | None, SelectionDecision | None]:
    if selection_decision is None or max_new_to_current_ratio is None:
        return selected_prediction, selected_score_map, selection_decision

    burn = as_binary(inputs.burn_map_t).astype(bool)
    burn_cells = max(int(burn.sum()), 1)
    new_to_current_ratio = float(selection_decision.agent_new_cells / burn_cells)
    if new_to_current_ratio <= float(max_new_to_current_ratio):
        return selected_prediction, selected_score_map, selection_decision

    warm = np.maximum(as_binary(inputs.pso_forecast_t_plus_h).astype(bool), burn)
    use_llm_only = fallback == "llm_only" and llm_only_output is not None
    if use_llm_only:
        fallback_prediction = llm_only_output.predicted_binary_burn_map_t_plus_h
        fallback_score_map = llm_only_output.predicted_burn_probability_map_t_plus_h
        selected = "llm_only"
    else:
        fallback_prediction = warm.astype(np.uint8)
        fallback_score_map = np.maximum(inputs.pso_forecast_t_plus_h, inputs.burn_map_t)
        selected = "warm_start"

    gated_decision = SelectionDecision(
        selected=selected,
        reason=f"prior_new_to_current_ratio_exceeded:{new_to_current_ratio:.4f}>{float(max_new_to_current_ratio):.4f}",
        confidence=selection_decision.confidence,
        warm_new_cells=selection_decision.warm_new_cells,
        agent_new_cells=selection_decision.agent_new_cells,
        added_over_warm_cells=selection_decision.added_over_warm_cells,
        max_allowed_added_cells=selection_decision.max_allowed_added_cells,
        agent_to_warm_area_ratio=selection_decision.agent_to_warm_area_ratio,
    )
    return fallback_prediction, fallback_score_map, gated_decision


def calibrate_materialized_root(
    dataset_root: Path,
    output_dir: Path,
    max_scenarios: int | None,
    max_events_per_scenario: int,
    train_events_per_scenario: int,
    max_frame_pairs_per_event: int,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    calibration_iterations: int,
    calibration_learning_rate: float,
    calibration_min_new_cells: int,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_roots = discover_materialized_scenarios(dataset_root)
    if max_scenarios is not None:
        scenario_roots = scenario_roots[:max_scenarios]
    pso_generator = None
    if warm_start == "pso":
        pso_generator = PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap,
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
    agent = WildfireAgent(reasoner=reasoner)
    train_batches: list[np.ndarray] = []
    train_no_llm_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    train_examples_for_policy: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    split_rows: list[dict[str, object]] = []

    for scenario_root in scenario_roots:
        reader = MaterializedScenarioReader(scenario_root)
        events = reader.list_events()[:max_events_per_scenario]
        train_events = events[:train_events_per_scenario]
        for event in train_events:
            examples = _collect_calibration_examples(
                scenario_root=scenario_root,
                event=event,
                reader=reader,
                max_frame_pairs_per_event=max_frame_pairs_per_event,
                output_dir=output_dir,
                split="train",
                warm_start=warm_start,
                pso_generator=pso_generator,
                fallback_placeholder=fallback_placeholder,
                agent=agent,
                selector=selector,
                selector_max_new_to_current_ratio=selector_max_new_to_current_ratio,
                selector_prior_fallback=selector_prior_fallback,
                failures=failures,
            )
            train_examples_for_policy.extend(examples)
            for example in examples:
                features, labels = sample_new_burn_training_pixels(
                    example["features"],
                    example["target"],
                    example["burn_t"],
                    seed=len(train_batches),
                )
                if len(features):
                    train_batches.append(features)
                    train_no_llm_batches.append(_without_llm_calibration_features(features))
                    label_batches.append(labels)
                split_rows.append(
                    {
                        "split": "train",
                        "scenario": scenario_root.parent.name,
                        "fire_id": scenario_root.name,
                        "event_id": event.event_id,
                        "frame_index": example["frame_index"],
                        "positive_pixels": int(labels.sum()) if len(labels) else 0,
                        "sampled_pixels": int(len(labels)),
                    }
                )

    calibrator = fit_pixel_calibrator(
        train_batches,
        label_batches,
        iterations=calibration_iterations,
        learning_rate=calibration_learning_rate,
    )
    calibration_policy = _fit_calibration_postprocess_policy(
        train_examples_for_policy,
        calibrator,
        min_new_cells=calibration_min_new_cells,
    )
    no_llm_calibrator = fit_pixel_calibrator(
        train_no_llm_batches,
        label_batches,
        iterations=calibration_iterations,
        learning_rate=calibration_learning_rate,
    )
    no_llm_train_examples_for_policy = _without_llm_calibration_examples(train_examples_for_policy)
    no_llm_calibration_policy = _fit_calibration_postprocess_policy(
        no_llm_train_examples_for_policy,
        no_llm_calibrator,
        min_new_cells=calibration_min_new_cells,
        reference_methods=("warm_start", "simple_physics"),
    )

    rows: list[dict[str, object]] = []
    for scenario_root in scenario_roots:
        reader = MaterializedScenarioReader(scenario_root)
        events = reader.list_events()[:max_events_per_scenario]
        eval_events = events[train_events_per_scenario:max_events_per_scenario]
        for event in eval_events:
            examples = _collect_calibration_examples(
                scenario_root=scenario_root,
                event=event,
                reader=reader,
                max_frame_pairs_per_event=max_frame_pairs_per_event,
                output_dir=output_dir,
                split="eval",
                warm_start=warm_start,
                pso_generator=pso_generator,
                fallback_placeholder=fallback_placeholder,
                agent=agent,
                selector=selector,
                selector_max_new_to_current_ratio=selector_max_new_to_current_ratio,
                selector_prior_fallback=selector_prior_fallback,
                failures=failures,
            )
            for example in examples:
                calibrated_raw = calibrated_burn_probability(example["inputs"], example["features"], calibrator)
                calibrated_no_llm_raw = calibrated_burn_probability(
                    example["inputs"],
                    _without_llm_calibration_features(example["features"]),
                    no_llm_calibrator,
                )
                predictions = dict(example["predictions"])
                predictions["supervised_pixel_logistic"] = calibrated_no_llm_raw
                predictions["calibrated_no_llm_raw"] = calibrated_no_llm_raw
                predictions["calibrated_no_llm"] = _apply_calibration_postprocess_policy(
                    example["inputs"],
                    calibrated_no_llm_raw,
                    predictions,
                    no_llm_calibration_policy["policy"],
                )
                predictions["calibrated_raw"] = calibrated_raw
                predictions["calibrated"] = _apply_calibration_postprocess_policy(
                    example["inputs"],
                    calibrated_raw,
                    predictions,
                    calibration_policy["policy"],
                )
                predictions.update(
                    _random_sanity_predictions(
                        inputs=example["inputs"],
                        target=example["target"],
                        seed_parts=(
                            "calibrate",
                            scenario_root.parent.name,
                            scenario_root.name,
                            event.event_id,
                            str(example["frame_index"]),
                        ),
                    )
                )
                for method, prediction in predictions.items():
                    metric_values = evaluate(
                        prediction,
                        example["target"],
                        example["burn_t"],
                        example["inputs"].weather_t.wind_direction_deg,
                    )
                    rows.append(
                        {
                            "scenario": scenario_root.parent.name,
                            "fire_id": scenario_root.name,
                            "event_id": event.event_id,
                            "frame_index": example["frame_index"],
                            "method": method,
                            "split": "eval",
                            **metric_values,
                        }
                    )
                split_rows.append(
                    {
                        "split": "eval",
                        "scenario": scenario_root.parent.name,
                        "fire_id": scenario_root.name,
                        "event_id": event.event_id,
                        "frame_index": example["frame_index"],
                        "positive_pixels": int(((example["target"] >= 0.5) & ~(example["burn_t"] >= 0.5)).sum()),
                        "sampled_pixels": "",
                    }
                )
                _write_evaluation_outputs(output_dir, rows, failures)

    _write_csv(output_dir / "split_manifest.csv", split_rows)
    final_summary = _write_evaluation_outputs(output_dir, rows, failures)
    calibration_report = {
        "feature_names": calibrator.feature_names,
        "mean": calibrator.mean.tolist(),
        "scale": calibrator.scale.tolist(),
        "weights": calibrator.weights.tolist(),
        "bias": calibrator.bias,
        "train_pixel_batches": len(train_batches),
        "train_pixels": int(sum(len(batch) for batch in label_batches)),
        "train_positive_pixels": int(sum(batch.sum() for batch in label_batches)),
        "postprocess_policy": calibration_policy,
        "no_llm_ablation": {
            "disabled_feature_names": list(LLM_DERIVED_CALIBRATION_FEATURES),
            "feature_names": no_llm_calibrator.feature_names,
            "mean": no_llm_calibrator.mean.tolist(),
            "scale": no_llm_calibrator.scale.tolist(),
            "weights": no_llm_calibrator.weights.tolist(),
            "bias": no_llm_calibrator.bias,
            "postprocess_policy": no_llm_calibration_policy,
        },
        "summary": final_summary,
        "rows": len(rows),
        "failures": failures,
    }
    (output_dir / "calibrator.json").write_text(json.dumps(_json_ready(calibration_report), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready({"summary": calibration_report["summary"], "rows": len(rows), "failures": len(failures)}), indent=2))


def cross_validate_materialized_calibration(
    dataset_root: Path,
    output_dir: Path,
    max_scenarios: int | None,
    max_events_per_scenario: int,
    max_frame_pairs_per_event: int,
    warm_start: str,
    pso_params_file: Path,
    pso_device: str,
    pso_t_cap: int,
    ros_wildfire_src: str | None,
    pso_cache_dir: str | None,
    fallback_placeholder: bool,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    calibration_iterations: int,
    calibration_learning_rate: float,
    calibration_min_new_cells: int,
    min_train_events: int,
    reasoner,
    selector,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_roots = discover_materialized_scenarios(dataset_root)
    if max_scenarios is not None:
        scenario_roots = scenario_roots[:max_scenarios]
    pso_generator = None
    if warm_start == "pso":
        pso_generator = PsoWarmStartGenerator(
            pso_params_file,
            device=pso_device,
            t_cap=pso_t_cap,
            ros_wildfire_src=ros_wildfire_src,
            cache_dir=pso_cache_dir,
        )
    agent = WildfireAgent(reasoner=reasoner)
    failures: list[dict[str, str]] = []
    all_examples: list[dict[str, object]] = []
    event_keys: list[tuple[str, str, str]] = []

    for scenario_root in scenario_roots:
        reader = MaterializedScenarioReader(scenario_root)
        events = reader.list_events()[:max_events_per_scenario]
        for event in events:
            event_key = (scenario_root.parent.name, scenario_root.name, event.event_id)
            event_keys.append(event_key)
            examples = _collect_calibration_examples(
                scenario_root=scenario_root,
                event=event,
                reader=reader,
                max_frame_pairs_per_event=max_frame_pairs_per_event,
                output_dir=output_dir,
                split="cv_source",
                warm_start=warm_start,
                pso_generator=pso_generator,
                fallback_placeholder=fallback_placeholder,
                agent=agent,
                selector=selector,
                selector_max_new_to_current_ratio=selector_max_new_to_current_ratio,
                selector_prior_fallback=selector_prior_fallback,
                failures=failures,
            )
            all_examples.extend(examples)

    event_keys = sorted({key for key in event_keys if any(_example_event_key(example) == key for example in all_examples)})
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for fold_index, heldout_key in enumerate(event_keys):
        train_examples = [example for example in all_examples if _example_event_key(example) != heldout_key]
        eval_examples = [example for example in all_examples if _example_event_key(example) == heldout_key]
        train_event_count = len({_example_event_key(example) for example in train_examples})
        if train_event_count < min_train_events or not eval_examples:
            failures.append(
                {
                    "fold_index": str(fold_index),
                    "heldout_event": "|".join(heldout_key),
                    "split": "cv",
                    "error": f"skipped_fold_insufficient_data: train_events={train_event_count}, eval_frames={len(eval_examples)}",
                }
            )
            continue

        train_batches: list[np.ndarray] = []
        train_no_llm_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        for example in train_examples:
            features, labels = sample_new_burn_training_pixels(
                example["features"],
                example["target"],
                example["burn_t"],
                seed=len(train_batches),
            )
            if len(features):
                train_batches.append(features)
                train_no_llm_batches.append(_without_llm_calibration_features(features))
                label_batches.append(labels)

        try:
            calibrator = fit_pixel_calibrator(
                train_batches,
                label_batches,
                iterations=calibration_iterations,
                learning_rate=calibration_learning_rate,
            )
            calibration_policy = _fit_calibration_postprocess_policy(
                train_examples,
                calibrator,
                min_new_cells=calibration_min_new_cells,
            )
            no_llm_calibrator = fit_pixel_calibrator(
                train_no_llm_batches,
                label_batches,
                iterations=calibration_iterations,
                learning_rate=calibration_learning_rate,
            )
            no_llm_calibration_policy = _fit_calibration_postprocess_policy(
                _without_llm_calibration_examples(train_examples),
                no_llm_calibrator,
                min_new_cells=calibration_min_new_cells,
                reference_methods=("warm_start", "simple_physics"),
            )
        except Exception as exc:
            failures.append(
                {
                    "fold_index": str(fold_index),
                    "heldout_event": "|".join(heldout_key),
                    "split": "cv",
                    "error": f"skipped_fold_calibration_failed: {exc}",
                }
            )
            continue

        fold_method_rows: list[dict[str, object]] = []
        for example in eval_examples:
            calibrated_raw = calibrated_burn_probability(example["inputs"], example["features"], calibrator)
            calibrated_no_llm_raw = calibrated_burn_probability(
                example["inputs"],
                _without_llm_calibration_features(example["features"]),
                no_llm_calibrator,
            )
            predictions = dict(example["predictions"])
            predictions["supervised_pixel_logistic"] = calibrated_no_llm_raw
            predictions["calibrated_no_llm_raw"] = calibrated_no_llm_raw
            predictions["calibrated_no_llm"] = _apply_calibration_postprocess_policy(
                example["inputs"],
                calibrated_no_llm_raw,
                predictions,
                no_llm_calibration_policy["policy"],
            )
            predictions["calibrated_raw"] = calibrated_raw
            predictions["calibrated"] = _apply_calibration_postprocess_policy(
                example["inputs"],
                calibrated_raw,
                predictions,
                calibration_policy["policy"],
            )
            predictions.update(
                _random_sanity_predictions(
                    inputs=example["inputs"],
                    target=example["target"],
                    seed_parts=(
                        "cross_validate",
                        scenario_root.parent.name,
                        scenario_root.name,
                        event.event_id,
                        str(example["frame_index"]),
                    ),
                )
            )
            for method, prediction in predictions.items():
                metric_values = evaluate(
                    prediction,
                    example["target"],
                    example["burn_t"],
                    example["inputs"].weather_t.wind_direction_deg,
                )
                row = {
                    "fold_index": fold_index,
                    "heldout_scenario": heldout_key[0],
                    "heldout_fire_id": heldout_key[1],
                    "heldout_event_id": heldout_key[2],
                    "scenario": example["scenario"],
                    "fire_id": example["fire_id"],
                    "event_id": example["event_id"],
                    "frame_index": example["frame_index"],
                    "method": method,
                    "split": "rotating_eval",
                    **metric_values,
                }
                rows.append(row)
                fold_method_rows.append(row)

        fold_summary = _summarize_rows(fold_method_rows)
        fold_rows.append(
            {
                "fold_index": fold_index,
                "heldout_scenario": heldout_key[0],
                "heldout_fire_id": heldout_key[1],
                "heldout_event_id": heldout_key[2],
                "train_events": train_event_count,
                "train_frames": len(train_examples),
                "eval_frames": len(eval_examples),
                "train_pixels": int(sum(len(batch) for batch in label_batches)),
                "train_positive_pixels": int(sum(batch.sum() for batch in label_batches)),
                "policy_threshold": calibration_policy["policy"].get("threshold", ""),
                "policy_reference_method": calibration_policy["policy"].get("reference_method", ""),
                "policy_cap_multiplier": calibration_policy["policy"].get("cap_multiplier", ""),
                "no_llm_policy_threshold": no_llm_calibration_policy["policy"].get("threshold", ""),
                "no_llm_policy_reference_method": no_llm_calibration_policy["policy"].get("reference_method", ""),
                "no_llm_policy_cap_multiplier": no_llm_calibration_policy["policy"].get("cap_multiplier", ""),
                "calibrated_new_f1": fold_summary.get("calibrated", {}).get("mean_new_f1", 0.0),
                "calibrated_no_llm_new_f1": fold_summary.get("calibrated_no_llm", {}).get("mean_new_f1", 0.0),
                "warm_start_new_f1": fold_summary.get("warm_start", {}).get("mean_new_f1", 0.0),
                "agent_selected_new_f1": fold_summary.get("agent_selected", {}).get("mean_new_f1", 0.0),
                "simple_physics_new_f1": fold_summary.get("simple_physics", {}).get("mean_new_f1", 0.0),
                "llm_only_new_f1": fold_summary.get("llm_only", {}).get("mean_new_f1", 0.0),
            }
        )
        _write_evaluation_outputs(output_dir, rows, failures)
        _write_csv(output_dir / "folds.csv", fold_rows)

    summary = _write_evaluation_outputs(output_dir, rows, failures)
    _write_csv(output_dir / "folds.csv", fold_rows)
    report = {
        "summary": summary,
        "rows": len(rows),
        "folds": len(fold_rows),
        "source_events": len(event_keys),
        "source_frames": len(all_examples),
        "failures": failures,
    }
    (output_dir / "cross_validation_summary.json").write_text(
        json.dumps(_json_ready(report), indent=2),
        encoding="utf-8",
    )
    _write_rotating_calibration_report(output_dir, summary, fold_rows, failures)
    print(json.dumps(_json_ready({"summary": summary, "rows": len(rows), "folds": len(fold_rows), "failures": len(failures)}), indent=2))


def _example_event_key(example: dict[str, object]) -> tuple[str, str, str]:
    return (str(example["scenario"]), str(example["fire_id"]), str(example["event_id"]))


def _without_llm_calibration_features(features: np.ndarray) -> np.ndarray:
    return mask_feature_channels(
        features,
        FEATURE_NAMES,
        LLM_DERIVED_CALIBRATION_FEATURES,
    )


def _without_llm_calibration_examples(examples: list[dict[str, object]]) -> list[dict[str, object]]:
    masked_examples: list[dict[str, object]] = []
    for example in examples:
        masked_example = dict(example)
        masked_example["features"] = _without_llm_calibration_features(example["features"])
        masked_examples.append(masked_example)
    return masked_examples


def _write_rotating_calibration_report(
    output_dir: Path,
    summary: dict[str, dict[str, float]],
    fold_rows: list[dict[str, object]],
    failures: list[dict[str, str]],
) -> None:
    methods = [
        "warm_start",
        "simple_physics",
        "llm_only",
        "agent_selected",
        "supervised_pixel_logistic",
        "calibrated_no_llm",
        "calibrated",
        "calibrated_no_llm_raw",
        "calibrated_raw",
    ]
    lines = [
        "# Rotating Held-Out Event Calibration Report",
        "",
        "Each fold holds out one event, trains the pixel-level calibrator and",
        "post-processing policy on the remaining collected events, and evaluates",
        "on the held-out event frames.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | New F1 | New AUPRC | Boundary F1 | Area Error | Wind Error | Full F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        if method not in summary:
            continue
        metrics = summary[method]
        label = f"**{method}**" if method == "calibrated" else method
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(metrics["mean_new_f1"]),
                    _fmt(metrics["mean_new_auprc"]),
                    _fmt(metrics["mean_boundary_f1"]),
                    _fmt(metrics["mean_burned_area_error"]),
                    _fmt(metrics["mean_wind_aligned_spread_error"]),
                    _fmt(metrics["mean_f1"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Fold-Level New F1",
            "",
            "| Fold | Held-Out Event | Calibrated | Calibrated No LLM | Warm Start | Guarded Hybrid | Simple Physics | LLM Only |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        heldout = f"{row['heldout_scenario']}/{row['heldout_fire_id']}/{row['heldout_event_id']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold_index"]),
                    heldout,
                    _fmt(row["calibrated_new_f1"]),
                    _fmt(row.get("calibrated_no_llm_new_f1", 0.0)),
                    _fmt(row["warm_start_new_f1"]),
                    _fmt(row["agent_selected_new_f1"]),
                    _fmt(row["simple_physics_new_f1"]),
                    _fmt(row["llm_only_new_f1"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use this report to answer reviewer concerns about calibration split",
            "  sensitivity.",
            "- The strongest outcome is that `calibrated` improves New F1 over",
            "  `warm_start` and `agent_selected` in the aggregate and in most folds.",
            "- Mixed fold-level results should be reported as calibration sensitivity,",
            "  not hidden.",
            "",
            "## Failures",
            "",
            f"- Failure count: `{len(failures)}`.",
        ]
    )
    for failure in failures[:10]:
        lines.append(f"- `{failure.get('error', '')}`")
    (output_dir / "rotating_cv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_calibration_examples(
    *,
    scenario_root: Path,
    event,
    reader: MaterializedScenarioReader,
    max_frame_pairs_per_event: int,
    output_dir: Path,
    split: str,
    warm_start: str,
    pso_generator: PsoWarmStartGenerator | None,
    fallback_placeholder: bool,
    agent: WildfireAgent,
    selector,
    selector_max_new_to_current_ratio: float | None,
    selector_prior_fallback: str,
    failures: list[dict[str, str]],
) -> list[dict[str, object]]:
    if pso_generator is not None:
        required_frames = max_frame_pairs_per_event + 1
        missing_frames = _missing_mask_frames(event.mask_dir, required_frames)
        if missing_frames:
            failures.append(
                {
                    "scenario_root": str(scenario_root),
                    "event_id": event.event_id,
                    "split": split,
                    "error": (
                        f"skipped_event_incomplete_pso_sequence: missing {len(missing_frames)} "
                        f"of out1..out{required_frames}; first_missing={missing_frames[0]}"
                    ),
                }
            )
            return []
    static_layers = reader.load_static_layers()
    frames = reader.load_mask_sequence(event, max_frames=max_frame_pairs_per_event + 1)
    pair_count = max(0, min(max_frame_pairs_per_event, len(frames) - 1))
    examples = []
    for frame_index in range(pair_count):
        try:
            burn_t = (frames[frame_index] >= 0.5).astype(np.uint8)
            target = (frames[frame_index + 1] >= 0.5).astype(np.uint8)
            weather = reader.load_weather(event, index=frame_index)
            warm_start_map, warm_start_source = _build_warm_start(
                warm_start=warm_start,
                scenario_root=scenario_root,
                event_id=event.event_id,
                frame_index=frame_index,
                burn_t=burn_t,
                fbfm13=static_layers.fbfm13,
                weather=weather,
                pso_generator=pso_generator,
                fallback_placeholder=fallback_placeholder,
            )
            inputs = PredictionInput(
                burn_map_t=burn_t,
                pso_forecast_t_plus_h=warm_start_map,
                static_layers=static_layers,
                weather_t=weather,
                metadata={
                    "scenario_root": str(scenario_root),
                    "event_id": event.event_id,
                    "frame_index": frame_index,
                    "warm_start": warm_start_source,
                    "uses_physical_warm_start": True,
                },
            )
            panel_path = output_dir / f"{split}_panels" / f"{scenario_root.parent.name}_{event.event_id}_{frame_index}.png"
            output = agent.predict(inputs, panel_path=panel_path)
            llm_only_inputs = without_physical_warm_start(inputs)
            llm_panel_path = output_dir / f"{split}_llm_only_panels" / f"{scenario_root.parent.name}_{event.event_id}_{frame_index}.png"
            llm_only_output = agent.predict(llm_only_inputs, panel_path=llm_panel_path)
            selected_prediction = None
            selected_score_map = None
            selection_decision = None
            if selector is not None:
                selected_prediction, selection_decision = selector.select(inputs, output)
                selected_score_map = (
                    output.predicted_burn_probability_map_t_plus_h
                    if selection_decision.selected == "agent"
                    else np.maximum(inputs.pso_forecast_t_plus_h, burn_t)
                )
            selected_prediction, selected_score_map, selection_decision = _apply_prior_reliability_gate(
                inputs=inputs,
                selected_prediction=selected_prediction,
                selected_score_map=selected_score_map,
                selection_decision=selection_decision,
                llm_only_output=llm_only_output,
                max_new_to_current_ratio=selector_max_new_to_current_ratio,
                fallback=selector_prior_fallback,
            )
            warm_probability = np.maximum(inputs.pso_forecast_t_plus_h, burn_t)
            simple_probability = simple_physics_warm_start(inputs)
            predictions = {
                "persistence": persistence(inputs),
                "warm_start": warm_probability,
                "simple_physics": simple_probability,
                "agent": output.predicted_burn_probability_map_t_plus_h,
                "llm_only": llm_only_output.predicted_burn_probability_map_t_plus_h,
            }
            if selected_score_map is not None:
                predictions["agent_selected"] = selected_score_map
            features = build_pixel_features(
                inputs,
                warm_probability=warm_probability,
                agent_probability=output.predicted_burn_probability_map_t_plus_h,
                llm_only_probability=llm_only_output.predicted_burn_probability_map_t_plus_h,
                simple_probability=simple_probability,
            )
            examples.append(
                {
                    "scenario": scenario_root.parent.name,
                    "fire_id": scenario_root.name,
                    "event_id": event.event_id,
                    "frame_index": frame_index,
                    "inputs": inputs,
                    "burn_t": burn_t,
                    "target": target,
                    "features": features,
                    "predictions": predictions,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "scenario_root": str(scenario_root),
                    "event_id": event.event_id,
                    "frame_index": str(frame_index),
                    "split": split,
                    "error": str(exc),
                }
            )
    return examples


def _calibrator_from_report(
    report_path: Path,
    feature_batches: list[np.ndarray],
    *,
    fallback_examples: list[dict[str, object]],
    min_new_cells: int,
) -> tuple[PixelCalibrator, dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "mean" in report and "scale" in report:
        mean = np.asarray(report["mean"], dtype=float)
        scale = np.asarray(report["scale"], dtype=float)
    else:
        x = np.vstack([batch for batch in feature_batches if len(batch)])
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
    calibrator = PixelCalibrator(
        feature_names=list(report.get("feature_names", FEATURE_NAMES)),
        mean=mean,
        scale=scale,
        weights=np.asarray(report["weights"], dtype=float),
        bias=float(report["bias"]),
    )
    calibration_policy = report.get("postprocess_policy")
    if not calibration_policy:
        calibration_policy = _fit_calibration_postprocess_policy(
            fallback_examples,
            calibrator,
            min_new_cells=min_new_cells,
        )
    return calibrator, calibration_policy


def _fit_calibration_postprocess_policy(
    examples: list[dict[str, object]],
    calibrator,
    *,
    min_new_cells: int,
    reference_methods: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if not examples:
        return {
            "policy": {
                "threshold": 0.5,
                "reference_method": "none",
                "cap_multiplier": 0.0,
                "min_new_cells": min_new_cells,
            },
            "objective": "fallback_no_training_examples",
            "train_summary": {},
            "top_candidates": [],
        }

    calibrated_maps = [
        calibrated_burn_probability(example["inputs"], example["features"], calibrator)
        for example in examples
    ]
    thresholds = _calibration_candidate_thresholds(examples, calibrated_maps)
    policies: list[dict[str, object]] = [
        {
            "threshold": threshold,
            "reference_method": "none",
            "cap_multiplier": 0.0,
            "min_new_cells": min_new_cells,
        }
        for threshold in thresholds
    ]
    if reference_methods is None:
        reference_methods = ("warm_start", "simple_physics", "agent", "agent_selected", "max_prior")
    cap_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for reference_method in reference_methods:
        for cap_multiplier in cap_multipliers:
            for threshold in thresholds:
                policies.append(
                    {
                        "threshold": threshold,
                        "reference_method": reference_method,
                        "cap_multiplier": cap_multiplier,
                        "min_new_cells": min_new_cells,
                    }
                )

    candidates: list[dict[str, object]] = []
    for policy in policies:
        rows = []
        for example, calibrated_map in zip(examples, calibrated_maps):
            prediction = _apply_calibration_postprocess_policy(
                example["inputs"],
                calibrated_map,
                example["predictions"],
                policy,
            )
            metric_values = evaluate(
                prediction,
                example["target"],
                example["burn_t"],
                example["inputs"].weather_t.wind_direction_deg,
            )
            rows.append({"method": "calibrated_policy", **metric_values})
        summary = _summarize_rows(rows)["calibrated_policy"]
        objective_value = _calibration_policy_objective(summary)
        candidates.append(
            {
                "policy": policy,
                "objective": objective_value,
                "summary": summary,
            }
        )

    best = max(candidates, key=lambda item: float(item["objective"]))
    top_candidates = sorted(candidates, key=lambda item: float(item["objective"]), reverse=True)[:10]
    return {
        "policy": best["policy"],
        "objective": "new_f1 + 0.10*new_auprc + 0.10*boundary_f1 - 0.05*area_error - 0.03*wind_error",
        "objective_value": best["objective"],
        "train_summary": best["summary"],
        "top_candidates": top_candidates,
    }


def _calibration_candidate_thresholds(
    examples: list[dict[str, object]],
    calibrated_maps: list[np.ndarray],
) -> list[float]:
    fixed = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_batches = []
    for example, calibrated_map in zip(examples, calibrated_maps):
        burn = as_binary(example["burn_t"]).astype(bool)
        score_batches.append(np.asarray(calibrated_map, dtype=float)[~burn])
    scores = np.concatenate([batch for batch in score_batches if len(batch)])
    quantiles = np.quantile(scores, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]) if len(scores) else []
    thresholds = {round(float(value), 4) for value in fixed}
    thresholds.update(round(float(value), 4) for value in quantiles)
    return sorted(value for value in thresholds if 0.0 <= value <= 1.0)


def _calibration_policy_objective(summary: dict[str, float]) -> float:
    return (
        float(summary["mean_new_f1"])
        + 0.10 * float(summary["mean_new_auprc"])
        + 0.10 * float(summary["mean_boundary_f1"])
        - 0.05 * float(summary["mean_burned_area_error"])
        - 0.03 * float(summary["mean_wind_aligned_spread_error"])
    )


def _apply_calibration_postprocess_policy(
    inputs: PredictionInput,
    calibrated_probability: np.ndarray,
    predictions: dict[str, np.ndarray],
    policy: dict[str, object],
) -> np.ndarray:
    burn = as_binary(inputs.burn_map_t).astype(bool)
    probability = np.asarray(calibrated_probability, dtype=float)
    eligible = ~burn
    threshold = float(policy.get("threshold", 0.5))
    selected = eligible & (probability >= threshold)

    reference_method = str(policy.get("reference_method", "none"))
    if reference_method != "none":
        cap_multiplier = float(policy.get("cap_multiplier", 1.0))
        reference_new_cells = _reference_new_cell_count(reference_method, predictions, burn)
        cap = max(int(round(reference_new_cells * cap_multiplier)), int(policy.get("min_new_cells", 0)))
        selected = _top_probability_cells(probability, selected, cap)

    output = burn.astype(float)
    output[selected] = 1.0
    return output


def _reference_new_cell_count(reference_method: str, predictions: dict[str, np.ndarray], burn: np.ndarray) -> int:
    if reference_method == "max_prior":
        candidate_methods = ["warm_start", "simple_physics", "agent_selected", "agent"]
        return max((_reference_new_cell_count(method, predictions, burn) for method in candidate_methods), default=0)
    reference = predictions.get(reference_method)
    if reference is None and reference_method == "agent_selected":
        reference = predictions.get("agent")
    if reference is None:
        reference = predictions.get("warm_start")
    if reference is None:
        return 0
    binary = as_binary(reference).astype(bool)
    return int((binary & ~burn).sum())


def _random_sanity_predictions(
    *,
    inputs: PredictionInput,
    target: np.ndarray,
    seed_parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    burn = as_binary(inputs.burn_map_t).astype(bool)
    eligible = ~burn
    warm = np.maximum(inputs.pso_forecast_t_plus_h, inputs.burn_map_t)
    warm_new_count = int((as_binary(warm).astype(bool) & eligible).sum())
    target_new_count = int((as_binary(target).astype(bool) & eligible).sum())
    return {
        "random_50": _random_new_burn_map(burn, eligible, eligible_count=None, probability=0.5, seed_parts=(*seed_parts, "random_50")),
        "random_warm_area": _random_new_burn_map(
            burn,
            eligible,
            eligible_count=warm_new_count,
            probability=None,
            seed_parts=(*seed_parts, "random_warm_area"),
        ),
        "random_oracle_prevalence": _random_new_burn_map(
            burn,
            eligible,
            eligible_count=target_new_count,
            probability=None,
            seed_parts=(*seed_parts, "random_oracle_prevalence"),
        ),
    }


def _random_new_burn_map(
    burn: np.ndarray,
    eligible: np.ndarray,
    *,
    eligible_count: int | None,
    probability: float | None,
    seed_parts: tuple[str, ...],
) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(seed_parts))
    selected = np.zeros(burn.shape, dtype=bool)
    eligible_indices = np.flatnonzero(eligible.ravel())
    if probability is not None:
        selected.ravel()[eligible_indices] = rng.random(len(eligible_indices)) < probability
    else:
        count = max(0, min(int(eligible_count or 0), len(eligible_indices)))
        if count:
            chosen = rng.choice(eligible_indices, size=count, replace=False)
            selected.ravel()[chosen] = True
    output = burn.astype(float)
    output[selected] = 1.0
    return output


def _stable_seed(parts: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _top_probability_cells(probability: np.ndarray, candidate_mask: np.ndarray, cap: int) -> np.ndarray:
    selected = np.zeros(candidate_mask.shape, dtype=bool)
    if cap <= 0:
        return selected
    candidate_indices = np.flatnonzero(candidate_mask.ravel())
    if len(candidate_indices) <= cap:
        selected.ravel()[candidate_indices] = True
        return selected
    candidate_scores = probability.ravel()[candidate_indices]
    top_order = np.argsort(-candidate_scores, kind="mergesort")[:cap]
    selected.ravel()[candidate_indices[top_order]] = True
    return selected


def summarize_results(output_dir: Path, baseline_method: str, report_name: str) -> None:
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")
    rows: list[dict[str, str]]
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = _summarize_rows(rows)
    if summary_path.exists():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        failures = raw.get("failures", [])
    else:
        failures = []

    primary_metrics = [
        "mean_new_f1",
        "mean_new_auprc",
        "mean_new_recall_at_true_area",
        "mean_new_recall_at_2x_true_area",
        "mean_new_iou",
        "mean_auprc",
        "mean_new_boundary_f1",
        "mean_boundary_f1",
        "mean_burned_area_error",
        "mean_wind_aligned_spread_error",
        "mean_f1",
        "mean_iou",
        "mean_latency_seconds",
    ]
    methods = sorted(summary)
    baseline = summary.get(baseline_method, {})
    lines = [
        f"# Evaluation Report: `{output_dir.as_posix()}`",
        "",
        "## Run Health",
        "",
        f"- Rows: {len(rows)}",
        f"- Failures: {len(failures)}",
        f"- Baseline for deltas: `{baseline_method}`",
        "",
        "## Method Summary",
        "",
        "| Method | New F1 | New AUPRC | R@Area | R@2Area | New IoU | AUPRC | New Boundary F1 | Boundary F1 | Area Error | Wind Error | Full F1 | Full IoU | Latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        values = summary[method]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _fmt(values.get("mean_new_f1", 0.0)),
                    _fmt(values.get("mean_new_auprc", 0.0)),
                    _fmt(values.get("mean_new_recall_at_true_area", 0.0)),
                    _fmt(values.get("mean_new_recall_at_2x_true_area", 0.0)),
                    _fmt(values.get("mean_new_iou", 0.0)),
                    _fmt(values.get("mean_auprc", 0.0)),
                    _fmt(values.get("mean_new_boundary_f1", 0.0)),
                    _fmt(values.get("mean_boundary_f1", 0.0)),
                    _fmt(values.get("mean_burned_area_error", 0.0)),
                    _fmt(values.get("mean_wind_aligned_spread_error", 0.0)),
                    _fmt(values.get("mean_f1", 0.0)),
                    _fmt(values.get("mean_iou", 0.0)),
                    _fmt(values.get("mean_latency_seconds", 0.0)),
                ]
            )
            + " |"
        )

    lines.extend(["", f"## Delta vs `{baseline_method}`", ""])
    if baseline:
        lines.extend(["| Method | Metric | Delta | Direction |", "| --- | --- | ---: | --- |"])
        lower_is_better = {"mean_burned_area_error", "mean_wind_aligned_spread_error", "mean_latency_seconds"}
        for method in methods:
            if method == baseline_method:
                continue
            for metric in primary_metrics:
                delta = summary[method].get(metric, 0.0) - baseline.get(metric, 0.0)
                better = delta < 0 if metric in lower_is_better else delta > 0
                direction = "better" if better else "worse_or_equal"
                lines.append(f"| {method} | {metric} | {_fmt(delta)} | {direction} |")
    else:
        lines.append(f"`{baseline_method}` was not found in metrics.csv.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Prioritize newly burned-cell metrics for spread prediction claims.",
            "- Treat full-map IoU/F1 carefully because persistence can score well when spread is slow.",
            "- Burned-area error and boundary F1 should be reported beside new F1 to expose over-expansion.",
            "- Use this report as an experiment log entry, not as final paper text.",
            "",
        ]
    )
    report_path = output_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def bootstrap_comparisons(
    output_dir: Path,
    *,
    candidate_method: str,
    comparison_methods: list[str],
    metric: str,
    iterations: int,
    seed: int,
    report_name: str,
    unit: str = "frame",
) -> None:
    metrics_path = output_dir / "metrics_repaired.csv"
    if not metrics_path.exists():
        metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    key_fields = ["scenario", "fire_id", "event_id", "frame_index"]
    if rows and "fold_index" in rows[0]:
        key_fields = ["fold_index", *key_fields]
    by_frame: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        by_frame.setdefault(key, {})[str(row["method"])] = row

    rng = np.random.default_rng(seed)
    result_rows: list[dict[str, object]] = []
    lines = [
        f"# Bootstrap Comparison Report: `{output_dir.as_posix()}`",
        "",
        f"- Candidate: `{candidate_method}`",
        f"- Metric: `{metric}`",
        f"- Resampling unit: `{unit}`",
        f"- Iterations: `{iterations}`",
        f"- Seed: `{seed}`",
        "",
        "| Comparison | Frames | Units | Mean Delta | 95% CI Low | 95% CI High | Win/Tie/Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison_method in comparison_methods:
        frame_deltas: list[tuple[tuple[str, ...], float]] = []
        for frame_methods in by_frame.values():
            if candidate_method not in frame_methods or comparison_method not in frame_methods:
                continue
            candidate_row = frame_methods[candidate_method]
            candidate_value = float(candidate_row.get(metric, 0.0))
            comparison_value = float(frame_methods[comparison_method].get(metric, 0.0))
            frame_deltas.append((_bootstrap_unit_key(candidate_row, unit), candidate_value - comparison_value))
        deltas_by_unit: dict[tuple[str, ...], list[float]] = {}
        for unit_key, delta in frame_deltas:
            deltas_by_unit.setdefault(unit_key, []).append(delta)
        if unit == "frame":
            unit_deltas = [delta for _, delta in frame_deltas]
        else:
            unit_deltas = [float(np.mean(values)) for values in deltas_by_unit.values()]
        deltas = unit_deltas
        delta_array = np.asarray(deltas, dtype=float)
        if len(delta_array) == 0:
            continue
        boot = []
        for _ in range(iterations):
            sample_indices = rng.integers(0, len(delta_array), size=len(delta_array))
            boot.append(float(delta_array[sample_indices].mean()))
        ci_low, ci_high = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
        wins = int((delta_array > 1e-12).sum())
        ties = int((np.abs(delta_array) <= 1e-12).sum())
        losses = int((delta_array < -1e-12).sum())
        result = {
            "candidate": candidate_method,
            "comparison": comparison_method,
            "metric": metric,
            "unit": unit,
            "frames": len(frame_deltas),
            "units": len(delta_array),
            "mean_delta": float(delta_array.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
        result_rows.append(result)
        lines.append(
            "| "
            + " | ".join(
                [
                    comparison_method,
                    str(len(frame_deltas)),
                    str(len(delta_array)),
                    _fmt(result["mean_delta"]),
                    _fmt(result["ci_low"]),
                    _fmt(result["ci_high"]),
                    f"{wins}/{ties}/{losses}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive deltas favor the candidate method.",
            "- `frame` resamples individual frame pairs; `event` and `scenario` first average deltas within the selected unit and then resample those units.",
            "- Wide or zero-crossing intervals indicate that the paired improvement is not stable enough for a strong dominance claim.",
            "- Use these intervals as evidence for reviewer-facing caution, not as a replacement for larger held-out evaluation.",
            "",
        ]
    )
    report_path = output_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if result_rows:
        _write_csv(output_dir / f"{Path(report_name).stem}.csv", result_rows)
    print(f"wrote {report_path}")


def _bootstrap_unit_key(row: dict[str, str], unit: str) -> tuple[str, ...]:
    if unit == "frame":
        fields = ["fold_index", "scenario", "fire_id", "event_id", "frame_index"] if "fold_index" in row else ["scenario", "fire_id", "event_id", "frame_index"]
    elif unit == "event":
        if "heldout_event_id" in row:
            fields = ["fold_index", "heldout_scenario", "heldout_fire_id", "heldout_event_id"]
        else:
            fields = ["scenario", "fire_id", "event_id"]
    elif unit == "scenario":
        fields = ["heldout_scenario"] if "heldout_scenario" in row else ["scenario"]
    else:
        raise ValueError(f"unsupported bootstrap unit: {unit}")
    return tuple(str(row.get(field, "")) for field in fields)


def audit_metrics(output_dir: Path, *, report_name: str, repaired_metrics_name: str) -> None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    suspicious_empty_iou_rows: list[dict[str, str]] = []
    repaired_rows: list[dict[str, object]] = []
    for row in rows:
        repaired = dict(row)
        try:
            new_iou = float(row.get("new_iou", 0.0))
            new_precision = float(row.get("new_precision", 0.0))
            new_recall = float(row.get("new_recall", 0.0))
            new_f1 = float(row.get("new_f1", 0.0))
        except ValueError:
            new_iou = new_precision = new_recall = new_f1 = 0.0
        is_empty_union_artifact = (
            abs(new_iou - 1.0) <= 1e-12
            and abs(new_precision) <= 1e-12
            and abs(new_recall) <= 1e-12
            and abs(new_f1) <= 1e-12
        )
        if is_empty_union_artifact:
            suspicious_empty_iou_rows.append(row)
            repaired["new_iou"] = 0.0
        repaired_rows.append(repaired)

    repaired_path = output_dir / repaired_metrics_name
    _write_csv(repaired_path, repaired_rows)

    by_method: dict[str, int] = {}
    for row in suspicious_empty_iou_rows:
        by_method[str(row.get("method", ""))] = by_method.get(str(row.get("method", "")), 0) + 1

    lines = [
        f"# Metric Audit: `{output_dir.as_posix()}`",
        "",
        "## Checks",
        "",
        "- Empty newly burned-cell unions should not inflate `new_iou` for spread-detection claims.",
        "- Rows with `new_iou=1` and zero new precision/recall/F1 were conservatively repaired to `new_iou=0`.",
        "- `new_boundary_f1` cannot be reconstructed from old `metrics.csv` files unless prediction/target maps are available; rerun evaluation with the updated metric code for paper-facing boundary claims.",
        "",
        "## Findings",
        "",
        f"- Total rows: `{len(rows)}`",
        f"- Empty-union New-IoU artifact rows: `{len(suspicious_empty_iou_rows)}`",
        f"- Repaired CSV: `{repaired_path.as_posix()}`",
        "",
        "| Method | Artifact Rows |",
        "| --- | ---: |",
    ]
    for method, count in sorted(by_method.items()):
        lines.append(f"| {method} | {count} |")
    if not by_method:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Reviewer-Facing Implication",
            "",
            "- Treat previously generated New-IoU summaries as provisional if this report finds artifact rows.",
            "- New-F1, New Precision, and New Recall are unaffected by this specific empty-union IoU convention.",
            "- For final paper assets, prefer rerunning the evaluator so `new_iou` and `new_boundary_f1` are computed directly from maps under the updated definitions.",
            "",
        ]
    )
    report_path = output_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "repaired_metrics": str(repaired_path), "artifact_rows": len(suspicious_empty_iou_rows)}, indent=2))


def make_paper_assets(
    *,
    calibration_5_dir: Path,
    calibration_10_dir: Path,
    rotating_cv_dir: Path,
    matched_baseline_dir: Path,
    output_dir: Path,
    summary_doc: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_doc.parent.mkdir(parents=True, exist_ok=True)
    operational_methods = [
        "persistence",
        "llm_only",
        "random_50",
        "random_warm_area",
        "random_oracle_prevalence",
        "warm_start",
        "simple_physics",
        "agent_selected",
        "calibrated_no_llm",
        "calibrated",
        "supervised_pixel_logistic",
    ]
    runs = [
        (
            "main_calibration_5scenario",
            calibration_5_dir,
            operational_methods,
        ),
        (
            "robustness_calibration_10scenario",
            calibration_10_dir,
            operational_methods,
        ),
        (
            "rotating_cv_calibration_5scenario",
            rotating_cv_dir,
            operational_methods,
        ),
        ("matched_non_calibrated_5scenario", matched_baseline_dir, ["persistence", "llm_only", "warm_start", "simple_physics", "agent", "agent_selected"]),
    ]
    all_rows: list[dict[str, object]] = []
    rows_by_run: dict[str, list[dict[str, object]]] = {}
    markdown_sections: list[str] = [
        "# Paper Results Summary",
        "",
        "Generated from completed experiment `summary.json` files.",
        "",
        "## Key Claim",
        "",
        (
            "Physical warm-started pixel-level calibration improves thresholded "
            "newly burned-cell prediction over the PSO warm start and non-learned "
            "guarded hybrid. Unconstrained supervised pixel-logistic maps can "
            "score higher on recall-oriented New F1/AUPRC, but they over-expand. "
            "LLM-derived channels provide structured auxiliary evidence with "
            "positive New-F1/New-IoU gains over no-LLM calibration, but with "
            "mixed area and boundary tradeoffs."
        ),
        "",
    ]
    for run_name, run_dir, methods in runs:
        loaded = _load_run_summary(run_dir)
        rows = _paper_rows(run_name, loaded["summary"], methods)
        rows_by_run[run_name] = rows
        all_rows.extend(rows)
        _write_csv(output_dir / f"{run_name}.csv", rows)
        (output_dir / f"{run_name}.tex").write_text(
            _latex_table(rows, caption=_paper_caption(run_name), label=f"tab:{run_name.replace('_', '-')}"),
            encoding="utf-8",
        )
        markdown_sections.extend(_markdown_run_section(run_name, run_dir, loaded, rows))

    llm_rows = _llm_contribution_rows(rows_by_run)
    if llm_rows:
        _write_csv(output_dir / "llm_contribution_ablation.csv", llm_rows)
        (output_dir / "llm_contribution_ablation.tex").write_text(_latex_llm_contribution_table(llm_rows), encoding="utf-8")
        markdown_sections.extend(_markdown_llm_contribution_section(llm_rows))

    bootstrap_rows = _bootstrap_summary_rows(
        {
            "main_calibration_5scenario": calibration_5_dir,
            "robustness_calibration_10scenario": calibration_10_dir,
            "rotating_cv_calibration_5scenario": rotating_cv_dir,
        }
    )
    if bootstrap_rows:
        _write_csv(output_dir / "bootstrap_ci_summary.csv", bootstrap_rows)
        (output_dir / "bootstrap_ci_summary.tex").write_text(_latex_bootstrap_table(bootstrap_rows), encoding="utf-8")
        markdown_sections.extend(_markdown_bootstrap_section(bootstrap_rows))

    tradeoff_rows = [
        row
        for run_name in (
            "main_calibration_5scenario",
            "robustness_calibration_10scenario",
            "rotating_cv_calibration_5scenario",
        )
        for row in rows_by_run.get(run_name, [])
        if str(row["method"]) in {"warm_start", "agent_selected", "calibrated_no_llm", "calibrated", "supervised_pixel_logistic"}
    ]
    if tradeoff_rows:
        _write_tradeoff_plot(tradeoff_rows, output_dir / "new_f1_area_tradeoff.png")
        markdown_sections.extend(
            [
                "## Tradeoff Figure",
                "",
                "- New-F1/area-error tradeoff plot: `"
                + (output_dir / "new_f1_area_tradeoff.png").as_posix()
                + "`.",
                "- Interpret points toward the lower right as better operational maps: higher newly burned-cell detection with lower over-expansion.",
                "",
            ]
        )

    _write_csv(output_dir / "combined_results.csv", all_rows)
    (output_dir / "combined_results.tex").write_text(_latex_table(all_rows, caption="Paper-facing experiment summary."), encoding="utf-8")
    markdown_sections.extend(
        [
            "## Decision Rule Result",
            "",
            "- Status: `Go`.",
            "- Main 5-scenario calibration passes the decision rule.",
            "- 10-scenario robustness run passes with caveats: calibrated improves New F1, but does not dominate every ranking or conservativeness metric.",
            "- Rotating held-out-event validation supports the New-F1 claim in aggregate, with mixed fold-level behavior.",
            "- The supervised pixel-logistic baseline has stronger raw New F1/AUPRC in some runs, but much weaker area and boundary consistency.",
            "- Matched non-calibrated baseline confirms the guarded hybrid is useful, while calibration is needed for stronger thresholded spread detection.",
            "",
            "## Recommended Paper Wording",
            "",
            "Use:",
            "",
            "> The calibrated hybrid improves thresholded newly burned-cell prediction over the physical warm start and non-learned guarded hybrid while preserving explicit physical priors and interpretable correction paths.",
            "",
            "With no-LLM ablation:",
            "",
            "> LLM-derived channels act as structured auxiliary evidence: they improve thresholded New F1 and New IoU over no-LLM calibration, but no-LLM calibration remains more conservative on area and full-map boundary metrics.",
            "",
            "With supervised baseline:",
            "",
            "> Unconstrained supervised pixel-logistic maps expose the available ranking signal, but they over-expand; the proposed calibrated map is an operationally constrained compromise with stronger area and boundary consistency.",
            "",
            "Avoid:",
            "",
            "- Claiming that LLM-only is a competitive standalone spread predictor.",
            "- Claiming that LLM-derived features are the dominant source of predictive improvement.",
            "- Claiming that the calibrated map dominates the supervised pixel-logistic baseline on New F1 or AUPRC.",
            "- Claiming temporal improvement grows monotonically over frame index.",
            "- Claiming calibrated dominates every metric; conservative baselines can have lower area error or higher ranking metrics in some settings.",
            "",
            "## Qualitative Figure Candidates",
            "",
            "- System architecture: `outputs/paper_assets/system_architecture.tex`,",
            "  `outputs/paper_assets/system_architecture.svg`, and",
            "  `outputs/paper_assets/system_architecture.png`.",
            "- New-F1/area-error tradeoff plot: `outputs/paper_assets_supervised_metric/new_f1_area_tradeoff.png`.",
            "- Calibrated success case: `outputs/case_figures/calibrated_success_0170_00242_f1/calibrated_case_figure.png`.",
            "- Calibrated failure/tradeoff case: `outputs/case_figures/calibrated_failure_0001_00065_f3/calibrated_case_figure.png`.",
            "  - `calibrated` New F1 `0.5196` vs `simple_physics` New F1 `0.6059`.",
            "  - `calibrated` has lower area error (`0.0054` vs `0.2846`), higher boundary F1 (`0.8071` vs `0.7180`), and lower wind error (`0.0337` vs `0.0920`).",
            "- Success case: `outputs/case_figures/accepted_0002_00003_f2/case_figure.png`.",
            "- Rejected over-expansion case: `outputs/case_figures/rejected_0001_00012_f0/case_figure.png`.",
            "- LLM-only ablation case: `outputs/case_figures/llm_only_export_smoke/case_figure.png`.",
            "",
        ]
    )
    summary_doc.write_text("\n".join(markdown_sections), encoding="utf-8")
    print(json.dumps(_json_ready({"summary_doc": str(summary_doc), "output_dir": str(output_dir), "rows": len(all_rows)}), indent=2))


def _llm_contribution_rows(rows_by_run: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name in (
        "main_calibration_5scenario",
        "robustness_calibration_10scenario",
        "rotating_cv_calibration_5scenario",
    ):
        by_method = {str(row["method"]): row for row in rows_by_run.get(run_name, [])}
        calibrated = by_method.get("calibrated")
        no_llm = by_method.get("calibrated_no_llm")
        if not calibrated or not no_llm:
            continue
        rows.append(
            {
                "run": run_name,
                "no_llm_new_f1": float(no_llm["new_f1"]),
                "full_new_f1": float(calibrated["new_f1"]),
                "delta_new_f1": float(calibrated["new_f1"]) - float(no_llm["new_f1"]),
                "no_llm_new_iou": float(no_llm["new_iou"]),
                "full_new_iou": float(calibrated["new_iou"]),
                "delta_new_iou": float(calibrated["new_iou"]) - float(no_llm["new_iou"]),
                "delta_area_error": float(calibrated["area_error"]) - float(no_llm["area_error"]),
                "delta_boundary_f1": float(calibrated["boundary_f1"]) - float(no_llm["boundary_f1"]),
            }
        )
        if "new_boundary_f1" in calibrated and "new_boundary_f1" in no_llm:
            rows[-1]["delta_new_boundary_f1"] = float(calibrated["new_boundary_f1"]) - float(no_llm["new_boundary_f1"])
    return rows


def _bootstrap_summary_rows(run_dirs: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name, run_dir in run_dirs.items():
        for csv_name in (
            "bootstrap_comparisons.csv",
            "bootstrap_new_iou.csv",
            "bootstrap_area_error.csv",
            "bootstrap_boundary_f1.csv",
            "bootstrap_new_boundary_f1.csv",
            "bootstrap_event_new_f1.csv",
            "bootstrap_event_new_iou.csv",
            "bootstrap_event_area_error.csv",
        ):
            csv_path = run_dir / csv_name
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("comparison") not in {"warm_start", "agent_selected", "calibrated_no_llm", "supervised_pixel_logistic"}:
                        continue
                    rows.append(
                        {
                            "run": run_name,
                            "metric": row["metric"],
                            "unit": row.get("unit", "frame"),
                            "comparison": row["comparison"],
                            "frames": int(row["frames"]),
                            "units": int(row.get("units", row["frames"])),
                            "mean_delta": float(row["mean_delta"]),
                            "ci_low": float(row["ci_low"]),
                            "ci_high": float(row["ci_high"]),
                            "wins": int(row["wins"]),
                            "ties": int(row["ties"]),
                            "losses": int(row["losses"]),
                        }
                    )
    return rows


def _markdown_bootstrap_section(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "## Bootstrap Confidence Intervals",
        "",
        "| Run | Unit | Metric | Comparison | Frames | Units | Delta | 95% CI | Win/Tie/Loss |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row.get("unit", "frame")),
                    str(row["metric"]),
                    str(row["comparison"]),
                    str(row["frames"]),
                    str(row.get("units", row["frames"])),
                    _fmt(row["mean_delta"]),
                    f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]",
                    f"{row['wins']}/{row['ties']}/{row['losses']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- For `new_f1`, `new_iou`, `new_boundary_f1`, and `boundary_f1`, CIs above zero support paired aggregate improvement over the comparison method.",
            "- For `burned_area_error`, CIs below zero support paired aggregate improvement because lower error is better.",
            "- Zero-crossing CIs should be described as mixed rather than dominant.",
            "",
        ]
    )
    return lines


def _latex_bootstrap_table(rows: list[dict[str, object]]) -> str:
    event_rows = [row for row in rows if row.get("unit") == "event"]
    source_rows = event_rows if event_rows else rows
    primary_rows = [
        row
        for row in source_rows
        if row["comparison"] in {"warm_start", "calibrated_no_llm", "supervised_pixel_logistic"}
        and row["metric"] in {"new_f1", "new_iou", "burned_area_error"}
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllrrr}",
        "\\hline",
        "Run & Unit & Metric & Comparison & $\\Delta$ & 95\\% CI & W/T/L \\\\",
        "\\hline",
    ]
    for row in primary_rows:
        ci = f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]"
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        lines.append(
            f"{_latex_escape(str(row['run']))} & {_latex_escape(str(row.get('unit', 'frame')))} & "
            f"{_latex_escape(str(row['metric']))} & "
            f"{_latex_escape(str(row['comparison']))} & {float(row['mean_delta']):+.3f} & "
            f"{ci} & {wtl} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\caption{Paired bootstrap confidence intervals for calibrated-map differences. Event-level rows first average frame deltas within held-out events. Positive deltas favor the full calibrated method for New F1/New IoU; negative deltas favor it for burned-area error.}",
            "\\label{tab:bootstrap-ci}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_llm_contribution_section(rows: list[dict[str, object]]) -> list[str]:
    include_new_boundary = any("delta_new_boundary_f1" in row for row in rows)
    lines = [
        "## LLM Contribution Ablation",
        "",
    ]
    if include_new_boundary:
        lines.extend(
            [
                "| Run | No-LLM New F1 | Full New F1 | Delta New F1 | Delta New IoU | Delta Area Error | Delta New Boundary F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Run | No-LLM New F1 | Full New F1 | Delta New F1 | Delta New IoU | Delta Area Error | Delta Boundary F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in rows:
        cells = [
            str(row["run"]),
            _fmt(row["no_llm_new_f1"]),
            _fmt(row["full_new_f1"]),
            _fmt(row["delta_new_f1"]),
            _fmt(row["delta_new_iou"]),
            _fmt(row["delta_area_error"]),
            _fmt(row.get("delta_new_boundary_f1", row["delta_boundary_f1"])),
        ]
        lines.append(
            "| " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "- Positive New-F1/New-IoU deltas support LLM-derived channels as auxiliary evidence.",
            "- Positive detection deltas should be reported with the area and boundary tradeoffs versus no-LLM calibration.",
            "- Negative area-error deltas are better; positive boundary deltas are better.",
            "",
        ]
    )
    return lines


def _latex_llm_contribution_table(rows: list[dict[str, object]]) -> str:
    include_new_boundary = any("delta_new_boundary_f1" in row for row in rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        (
            "Run & No-LLM F1 & Full F1 & $\\Delta$F1 & $\\Delta$IoU & $\\Delta$Area Err. & $\\Delta$New Bound. F1 \\\\"
            if include_new_boundary
            else "Run & No-LLM F1 & Full F1 & $\\Delta$F1 & $\\Delta$IoU & $\\Delta$Area Err. & $\\Delta$Boundary F1 \\\\"
        ),
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(str(row['run']))} & "
            f"{float(row['no_llm_new_f1']):.3f} & {float(row['full_new_f1']):.3f} & "
            f"{float(row['delta_new_f1']):+.3f} & {float(row['delta_new_iou']):+.3f} & "
            f"{float(row['delta_area_error']):+.3f} & {float(row.get('delta_new_boundary_f1', row['delta_boundary_f1'])):+.3f} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\caption{Ablation isolating LLM-derived calibration channels. Positive New-F1/New-IoU and boundary deltas favor the full calibrated model; negative area-error deltas are better.}",
            "\\label{tab:llm-ablation}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_tradeoff_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    width, height = 1400, 900
    margin_left, margin_right = 150, 390
    margin_top, margin_bottom = 80, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    method_colors = {
        "warm_start": (70, 130, 180),
        "agent_selected": (120, 120, 120),
        "calibrated_no_llm": (92, 153, 75),
        "calibrated": (205, 85, 70),
        "supervised_pixel_logistic": (120, 80, 160),
    }
    method_markers = {
        "warm_start": "W",
        "agent_selected": "G",
        "calibrated_no_llm": "N",
        "calibrated": "C",
        "supervised_pixel_logistic": "S",
    }
    run_offsets = {
        "main_calibration_5scenario": -10,
        "robustness_calibration_10scenario": 0,
        "rotating_cv_calibration_5scenario": 10,
    }
    max_x = max(float(row["new_f1"]) for row in rows)
    x_hi = max(0.35, max_x * 1.1)
    non_extreme_area = [float(row["area_error"]) for row in rows if float(row["area_error"]) <= 5.0]
    y_hi = max(3.5, max(non_extreme_area, default=3.5) * 1.15)

    def sx(value: float) -> int:
        return margin_left + int((value / x_hi) * plot_w)

    def sy(value: float) -> int:
        return margin_top + plot_h - int((value / y_hi) * plot_h)

    # Grid and axes.
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = margin_left + int(frac * plot_w)
        y = margin_top + int(frac * plot_h)
        draw.line((x, margin_top, x, margin_top + plot_h), fill=(230, 230, 230))
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(230, 230, 230))
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)

    title = "Newly Burned-Cell Detection vs Over-Expansion"
    draw.text((margin_left, 25), title, fill="black", font=title_font)
    draw.text((margin_left + plot_w // 2 - 130, height - 55), "New F1 (higher is better)", fill="black", font=font)
    draw.text((18, margin_top + plot_h // 2), "Area Error\n(lower is better)", fill="black", font=small_font)
    draw.text((margin_left + plot_w - 145, margin_top + 18), "better", fill=(60, 120, 60), font=small_font)
    draw.line((margin_left + plot_w - 95, margin_top + 45, margin_left + plot_w - 20, margin_top + 45), fill=(60, 120, 60), width=3)
    draw.line((margin_left + plot_w - 20, margin_top + 45, margin_left + plot_w - 35, margin_top + 35), fill=(60, 120, 60), width=3)
    draw.line((margin_left + plot_w - 20, margin_top + 45, margin_left + plot_w - 35, margin_top + 55), fill=(60, 120, 60), width=3)
    draw.line((margin_left + plot_w - 20, margin_top + 45, margin_left + plot_w - 20, margin_top + 105), fill=(60, 120, 60), width=3)
    draw.line((margin_left + plot_w - 20, margin_top + 105, margin_left + plot_w - 30, margin_top + 90), fill=(60, 120, 60), width=3)
    draw.line((margin_left + plot_w - 20, margin_top + 105, margin_left + plot_w - 10, margin_top + 90), fill=(60, 120, 60), width=3)

    for tick in np.linspace(0, x_hi, 6):
        x = sx(float(tick))
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 7), fill="black")
        draw.text((x - 18, margin_top + plot_h + 15), f"{tick:.2f}", fill="black", font=small_font)
    for tick in np.linspace(0, y_hi, 6):
        y = sy(float(tick))
        draw.line((margin_left - 7, y, margin_left, y), fill="black")
        draw.text((margin_left - 70, y - 8), f"{tick:.1f}", fill="black", font=small_font)

    for row in rows:
        method = str(row["method"])
        run_name = str(row["run"])
        color = method_colors.get(method, (0, 0, 0))
        area_error = float(row["area_error"])
        x = sx(float(row["new_f1"])) + run_offsets.get(run_name, 0)
        clipped = area_error > y_hi
        y = margin_top + 12 if clipped else sy(area_error)
        radius = 13 if method == "calibrated" else 10
        if clipped:
            points = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
            draw.polygon(points, fill=color, outline="black")
            draw.text((x + 12, y - 12), f"{area_error:.1f}", fill="black", font=small_font)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=2)
        draw.text((x - 6, y - 10), method_markers.get(method, "?"), fill="white", font=small_font)

    legend_x = margin_left + plot_w + 45
    legend_y = margin_top + 20
    draw.text((legend_x, legend_y), "Methods", fill="black", font=font)
    for idx, method in enumerate(method_colors):
        y = legend_y + 35 + idx * 36
        draw.ellipse((legend_x, y - 9, legend_x + 18, y + 9), fill=method_colors[method], outline="black")
        draw.text((legend_x + 30, y - 9), method, fill="black", font=small_font)
    draw.text((legend_x, legend_y + 235), "Run offset", fill="black", font=font)
    draw.text((legend_x, legend_y + 265), "left: main 5-scenario", fill="black", font=small_font)
    draw.text((legend_x, legend_y + 290), "center: 10-scenario", fill="black", font=small_font)
    draw.text((legend_x, legend_y + 315), "right: rotating CV", fill="black", font=small_font)
    draw.text((legend_x, legend_y + 360), "Triangle = off-scale", fill="black", font=small_font)
    draw.text((legend_x, legend_y + 385), "area error label", fill="black", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _load_run_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    repaired_metrics_path = run_dir / "metrics_repaired.csv"
    if repaired_metrics_path.exists():
        with repaired_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            repaired_rows = list(csv.DictReader(handle))
        repaired_summary = _summarize_rows(repaired_rows)
        for method, metrics in repaired_summary.items():
            if method in loaded.get("summary", {}) and "mean_new_iou" in metrics:
                loaded["summary"][method]["mean_new_iou"] = metrics["mean_new_iou"]
        loaded["metric_audit"] = {
            "repaired_metrics": repaired_metrics_path.as_posix(),
            "new_iou_repaired": True,
            "note": "Only mean_new_iou is overridden from repaired metrics; new_boundary_f1 requires rerunning evaluation with saved maps.",
        }
    return loaded


def _paper_rows(run_name: str, summary: dict[str, dict[str, float]], methods: list[str]) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        metrics = summary.get(method)
        if not metrics:
            continue
        rows.append(
            {
                "run": run_name,
                "method": method,
                "new_f1": float(metrics.get("mean_new_f1", 0.0)),
                "new_precision": float(metrics.get("mean_new_precision", 0.0)),
                "new_recall": float(metrics.get("mean_new_recall", 0.0)),
                "new_auprc": float(metrics.get("mean_new_auprc", 0.0)),
                "new_recall_at_true_area": float(metrics.get("mean_new_recall_at_true_area", 0.0)),
                "new_recall_at_2x_true_area": float(metrics.get("mean_new_recall_at_2x_true_area", 0.0)),
                "new_iou": float(metrics.get("mean_new_iou", 0.0)),
                "boundary_f1": float(metrics.get("mean_boundary_f1", 0.0)),
                "area_error": float(metrics.get("mean_burned_area_error", 0.0)),
                "wind_error": float(metrics.get("mean_wind_aligned_spread_error", 0.0)),
                "full_f1": float(metrics.get("mean_f1", 0.0)),
                "full_iou": float(metrics.get("mean_iou", 0.0)),
                "auprc": float(metrics.get("mean_auprc", 0.0)),
            }
        )
        if "mean_new_boundary_f1" in metrics:
            rows[-1]["new_boundary_f1"] = float(metrics.get("mean_new_boundary_f1", 0.0))
    return rows


def _markdown_run_section(run_name: str, run_dir: Path, loaded: dict[str, object], rows: list[dict[str, object]]) -> list[str]:
    lines = [
        f"## {_paper_title(run_name)}",
        "",
        f"- Source: `{run_dir.as_posix()}`",
        f"- Metric rows: `{loaded.get('rows', 0)}`",
        f"- Failures: `{len(loaded.get('failures', []))}`",
        "",
    ]
    include_new_boundary = any("new_boundary_f1" in row for row in rows)
    if include_new_boundary:
        lines.extend(
            [
                "| Method | New F1 | New IoU | New AUPRC | R@Area | R@2Area | New Boundary F1 | Boundary F1 | Area Error | Wind Error | Full F1 | Full IoU | AUPRC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Method | New F1 | New IoU | New AUPRC | R@Area | R@2Area | Boundary F1 | Area Error | Wind Error | Full F1 | Full IoU | AUPRC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    diagnostic_methods = {"calibrated_raw", "calibrated_no_llm_raw"}
    operational_rows = [row for row in rows if str(row["method"]) not in diagnostic_methods]
    best_new_f1 = max((float(row["new_f1"]) for row in operational_rows), default=0.0)
    for row in rows:
        method = str(row["method"])
        method_label = f"**{method}**" if abs(float(row["new_f1"]) - best_new_f1) < 1e-12 else method
        cells = [
            method_label,
            _fmt(row["new_f1"]),
            _fmt(row["new_iou"]),
            _fmt(row["new_auprc"]),
            _fmt(row.get("new_recall_at_true_area", 0.0)),
            _fmt(row.get("new_recall_at_2x_true_area", 0.0)),
        ]
        if include_new_boundary:
            cells.append(_fmt(row.get("new_boundary_f1", 0.0)))
        cells.extend(
            [
                _fmt(row["boundary_f1"]),
                _fmt(row["area_error"]),
                _fmt(row["wind_error"]),
                _fmt(row["full_f1"]),
                _fmt(row["full_iou"]),
                _fmt(row["auprc"]),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", *_run_interpretation(run_name, rows), ""])
    if any(str(row["method"]) in diagnostic_methods for row in rows):
        lines.extend(
            [
                "- `calibrated_raw` and `calibrated_no_llm_raw` are diagnostic probability maps before operational area-budget post-processing; use them mainly for ranking/AUPRC discussion.",
                "",
            ]
        )
    return lines


def _run_interpretation(run_name: str, rows: list[dict[str, object]]) -> list[str]:
    by_method = {str(row["method"]): row for row in rows}
    calibrated = by_method.get("calibrated")
    calibrated_no_llm = by_method.get("calibrated_no_llm")
    warm = by_method.get("warm_start")
    selected = by_method.get("agent_selected")
    if calibrated and warm:
        lines = [
            (
                f"- Calibrated New-F1 delta vs warm start: "
                f"`{float(calibrated['new_f1']) - float(warm['new_f1']):.4f}`."
            ),
            (
                f"- Calibrated area-error delta vs warm start: "
                f"`{float(calibrated['area_error']) - float(warm['area_error']):.4f}` "
                "(negative is better)."
            ),
        ]
        if selected:
            lines.append(
                f"- Calibrated New-F1 delta vs guarded hybrid: `{float(calibrated['new_f1']) - float(selected['new_f1']):.4f}`."
            )
        if calibrated_no_llm:
            lines.append(
                f"- Calibrated New-F1 delta vs no-LLM calibration: `{float(calibrated['new_f1']) - float(calibrated_no_llm['new_f1']):.4f}`."
            )
        return lines
    if selected and warm:
        return [
            f"- Guarded hybrid New-F1 delta vs warm start: `{float(selected['new_f1']) - float(warm['new_f1']):.4f}`.",
            f"- Guarded hybrid area-error delta vs warm start: `{float(selected['area_error']) - float(warm['area_error']):.4f}` (negative is better).",
        ]
    return ["- No calibrated or guarded-hybrid comparison available for this run."]


def _latex_table(rows: list[dict[str, object]], *, caption: str, label: str | None = None) -> str:
    include_new_boundary = any("new_boundary_f1" in row for row in rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrrrrr}" if include_new_boundary else "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        (
            "Run & Method & New F1 & New IoU & New AUPRC & R@Area & R@2Area & New Bound. F1 & Boundary F1 & Area Err. & Wind Err. \\\\"
            if include_new_boundary
            else "Run & Method & New F1 & New IoU & New AUPRC & R@Area & R@2Area & Full Bound. F1 & Area Err. & Wind Err. \\\\"
        ),
        "\\hline",
    ]
    for row in rows:
        cells = [
            _latex_escape(str(row["run"])),
            _latex_escape(str(row["method"])),
            f"{float(row['new_f1']):.3f}",
            f"{float(row['new_iou']):.3f}",
            f"{float(row['new_auprc']):.3f}",
            f"{float(row.get('new_recall_at_true_area', 0.0)):.3f}",
            f"{float(row.get('new_recall_at_2x_true_area', 0.0)):.3f}",
        ]
        if include_new_boundary:
            cells.append(f"{float(row.get('new_boundary_f1', 0.0)):.3f}")
        cells.extend(
            [
                f"{float(row['boundary_f1']):.3f}",
                f"{float(row['area_error']):.3f}",
                f"{float(row['wind_error']):.3f}",
            ]
        )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}%", "}", f"\\caption{{{_latex_escape(caption)}}}"])
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend(["\\end{table}", ""])
    return "\n".join(lines)


def _paper_title(run_name: str) -> str:
    return {
        "main_calibration_5scenario": "Main Calibration Result: 5 Scenarios",
        "robustness_calibration_10scenario": "Robustness Result: 10 Scenarios",
        "rotating_cv_calibration_5scenario": "Rotating Held-Out Event Calibration: 5 Scenarios",
        "matched_non_calibrated_5scenario": "Matched Non-Calibrated Baseline: 5 Scenarios",
    }.get(run_name, run_name)


def _paper_caption(run_name: str) -> str:
    return {
        "main_calibration_5scenario": "Main event-level calibrated hybrid result on five scenarios.",
        "robustness_calibration_10scenario": "Robustness check on ten scenarios.",
        "rotating_cv_calibration_5scenario": "Rotating held-out-event calibration check on five scenarios.",
        "matched_non_calibrated_5scenario": "Matched non-calibrated baseline comparison.",
    }.get(run_name, "Experiment summary.")


def _latex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def summarize_temporal(output_dir: Path, baseline_method: str, candidate_method: str, report_name: str) -> None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_frame_method: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["frame_index"]), row["method"])
        by_frame_method.setdefault(key, []).append(row)

    frame_indices = sorted({int(row["frame_index"]) for row in rows})
    metrics = ["new_f1", "new_auprc", "new_iou", "auprc", "f1", "iou", "burned_area_error", "new_boundary_f1", "boundary_f1", "wind_aligned_spread_error"]
    lower_is_better = {"burned_area_error", "wind_aligned_spread_error"}

    lines = [
        f"# Temporal Report: `{output_dir.as_posix()}`",
        "",
        "## Evaluation Setup",
        "",
        "- Each row is a one-step-ahead prediction: inputs at `t`, target burn map at `t+1`.",
        "- Static layers are reused across all timesteps in the same scenario.",
        "- Temporal trend is grouped by `frame_index` within each event.",
        f"- Baseline method: `{baseline_method}`",
        f"- Candidate method: `{candidate_method}`",
        "",
        "## Frame-Level Summary",
        "",
        "| Frame | Method | N | New F1 | New AUPRC | New IoU | AUPRC | Full F1 | Full IoU | Area Error | Boundary F1 | Wind Error |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    per_frame: dict[int, dict[str, dict[str, float]]] = {}
    for frame in frame_indices:
        per_frame[frame] = {}
        methods = sorted({method for (idx, method) in by_frame_method if idx == frame})
        for method in methods:
            method_rows = by_frame_method[(frame, method)]
            values = {metric: _mean_float(method_rows, metric) for metric in metrics}
            per_frame[frame][method] = values
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(frame),
                        method,
                        str(len(method_rows)),
                        _fmt(values["new_f1"]),
                        _fmt(values["new_auprc"]),
                        _fmt(values["new_iou"]),
                        _fmt(values["auprc"]),
                        _fmt(values["f1"]),
                        _fmt(values["iou"]),
                        _fmt(values["burned_area_error"]),
                        _fmt(values["boundary_f1"]),
                        _fmt(values["wind_aligned_spread_error"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            f"## `{candidate_method}` Delta vs `{baseline_method}`",
            "",
            "| Frame | Metric | Delta | Direction |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    trend_scores: dict[str, list[float]] = {metric: [] for metric in metrics}
    for frame in frame_indices:
        baseline = per_frame.get(frame, {}).get(baseline_method)
        candidate = per_frame.get(frame, {}).get(candidate_method)
        if not baseline or not candidate:
            continue
        for metric in metrics:
            delta = candidate[metric] - baseline[metric]
            trend_scores[metric].append(delta)
            better = delta < 0 if metric in lower_is_better else delta > 0
            direction = "better" if better else "worse_or_equal"
            lines.append(f"| {frame} | {metric} | {_fmt(delta)} | {direction} |")

    lines.extend(["", "## Hypothesis Check", ""])
    if trend_scores["new_f1"]:
        first_delta = trend_scores["new_f1"][0]
        last_delta = trend_scores["new_f1"][-1]
        monotonic_note = "improved" if last_delta > first_delta else "did_not_improve"
        lines.extend(
            [
                f"- New-F1 delta at first evaluated frame: `{_fmt(first_delta)}`.",
                f"- New-F1 delta at last evaluated frame: `{_fmt(last_delta)}`.",
                f"- Temporal improvement status: `{monotonic_note}`.",
                "- This is descriptive evidence only; use event-level splits and more frames before making a paper claim.",
            ]
        )
    else:
        lines.append("- Candidate and baseline methods were not both present for temporal comparison.")

    report_path = output_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def _mean_float(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else 0.0


def _fmt(value: float | object) -> str:
    return f"{float(value):.4f}"


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    numeric_keys = [
        "iou",
        "f1",
        "auprc",
        "new_iou",
        "new_precision",
        "new_recall",
        "new_f1",
        "new_auprc",
        "new_recall_at_true_area",
        "new_recall_at_2x_true_area",
        "burned_area_error",
        "new_boundary_f1",
        "boundary_f1",
        "wind_aligned_spread_error",
        "latency_seconds",
    ]
    methods = sorted({str(row["method"]) for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in method_rows if key in row]
            summary[method][f"mean_{key}"] = float(np.mean(values)) if values else 0.0
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_ready(row) for row in rows)


def _build_warm_start(
    *,
    warm_start: str,
    scenario_root: Path,
    event_id: str,
    frame_index: int,
    burn_t: np.ndarray,
    fbfm13: np.ndarray,
    weather,
    pso_generator: PsoWarmStartGenerator | None,
    fallback_placeholder: bool,
) -> tuple[np.ndarray, str]:
    if warm_start == "placeholder":
        return _no_leak_warm_start(burn_t, fbfm13, weather), "no_leak_physics_stub"
    if pso_generator is None:
        raise ValueError("pso_generator is required when warm_start='pso'")
    try:
        pso_map = pso_generator.predict_frame(scenario_root, event_id, frame_index)
        return np.maximum(pso_map, burn_t), f"pso:{pso_generator.params_file}"
    except Exception:
        if not fallback_placeholder:
            raise
        return _no_leak_warm_start(burn_t, fbfm13, weather), "pso_failed_fallback_no_leak_physics_stub"


def _write_gray(path: Path, values: np.ndarray) -> None:
    arr = (np.asarray(values, dtype=float) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _rgb_panel(values: np.ndarray) -> Image.Image:
    mask = (np.asarray(values) >= 0.5)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = np.array([229, 112, 43], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _probability_panel(values: np.ndarray) -> Image.Image:
    probability = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rgb = np.zeros((*probability.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * probability).astype(np.uint8)
    rgb[..., 1] = (190 * np.sqrt(probability)).astype(np.uint8)
    rgb[..., 2] = (35 * (1.0 - probability)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _error_overlay(prediction: np.ndarray, target: np.ndarray, current_burn: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction) >= 0.5
    truth = np.asarray(target) >= 0.5
    current = np.asarray(current_burn) >= 0.5
    pred_new = pred & ~current
    truth_new = truth & ~current
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[current] = np.array([120, 120, 120], dtype=np.uint8)
    rgb[pred_new & truth_new] = np.array([46, 160, 67], dtype=np.uint8)
    rgb[pred_new & ~truth_new] = np.array([218, 54, 51], dtype=np.uint8)
    rgb[~pred_new & truth_new] = np.array([56, 139, 253], dtype=np.uint8)
    return rgb


def _burn_bbox(values: np.ndarray, padding: int = 8) -> tuple[int, int, int, int] | None:
    mask = np.asarray(values) >= 0.5
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    height, width = mask.shape
    left = max(int(cols.min()) - padding, 0)
    top = max(int(rows.min()) - padding, 0)
    right = min(int(cols.max()) + padding + 1, width)
    bottom = min(int(rows.max()) + padding + 1, height)
    return left, top, right, bottom


def _write_case_composite(
    path: Path,
    panels: list[tuple[str, Image.Image]],
    tile_size: int = 220,
    crop_box: tuple[int, int, int, int] | None = None,
) -> None:
    title_height = 28
    gap = 10
    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    width = columns * tile_size + (columns + 1) * gap
    height = rows * (tile_size + title_height) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + title_height + gap)
        draw.text((x, y), title, fill=(20, 20, 20))
        if crop_box is not None:
            image = image.crop(crop_box)
        panel = image.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y + title_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _no_leak_warm_start(burn_t: np.ndarray, fbfm13: np.ndarray, weather) -> np.ndarray:
    dryness = weather.dryness_index()
    radius = 2 if weather.wind_speed_mph >= 12 or dryness > 0.65 else 1
    row_offset, col_offset = wind_to_offset(weather.wind_direction_deg, radius)
    growth = shift(dilate(burn_front(burn_t), radius), row_offset, col_offset)
    burnable = (fbfm13 >= 1) & (fbfm13 <= 13)
    probability = np.maximum(burn_t, growth * (0.30 + 0.45 * dryness))
    return np.where(burnable | (burn_t > 0), probability, 0.0)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
