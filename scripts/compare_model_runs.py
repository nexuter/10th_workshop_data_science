from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METHODS = [
    "warm_start",
    "agent",
    "llm_only",
    "agent_selected",
    "calibrated_no_llm",
    "calibrated",
    "supervised_pixel_logistic",
    "random_warm_area",
    "random_oracle_prevalence",
]

DEFAULT_METRICS = [
    "mean_new_f1",
    "mean_new_iou",
    "mean_new_auprc",
    "mean_burned_area_error",
    "mean_boundary_f1",
    "mean_new_boundary_f1",
    "mean_wind_aligned_spread_error",
]


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be formatted as label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path)


def _load_summary(path: Path) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return dict(data.get("summary", {}))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_rows(runs: list[tuple[str, Path]], methods: list[str], metrics: list[str]) -> list[dict[str, str]]:
    loaded = [(label, _load_summary(path)) for label, path in runs]
    rows: list[dict[str, str]] = []
    for method in methods:
        for metric in metrics:
            row: dict[str, str] = {"method": method, "metric": metric}
            values: list[float] = []
            labels: list[str] = []
            for label, summary in loaded:
                raw = summary.get(method, {}).get(metric)
                row[label] = _fmt(raw)
                if raw is not None:
                    values.append(float(raw))
                    labels.append(label)
            if len(values) >= 2:
                row["range"] = f"{max(values) - min(values):.4f}"
                row["best_value"] = f"{max(values):.4f}"
                row["best_model"] = labels[values.index(max(values))]
            else:
                row["range"] = ""
                row["best_value"] = ""
                row["best_model"] = ""
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], run_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["method", "metric", *run_labels, "range", "best_model"]
    lines = ["# Model Run Comparison", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- Compare full large runs only when all runs use the same scenario/event/frame limits.",
            "- Higher is better for F1, IoU, AUPRC, boundary metrics, and wind-aligned agreement.",
            "- Lower is better for burned-area error, so inspect that metric separately rather than using `best_model` mechanically.",
            "- LLM-derived gains should be interpreted against `calibrated_no_llm`; if the difference is small, the model effect is auxiliary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    rows = build_rows(args.run, args.methods, args.metrics)
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, [label for label, _path in args.run])


if __name__ == "__main__":
    main()
