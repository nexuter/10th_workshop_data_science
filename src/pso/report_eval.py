import argparse
import csv
from pathlib import Path
from collections import defaultdict
import statistics


def load_scenario_results(eval_dir: Path) -> dict:
    """Load evaluation results from all scenario folders.
    
    Returns dict: {scenario_name: {metrics: stats, sample_count: int}}
    """
    results = {}
    
    for scenario_folder in sorted(eval_dir.iterdir()):
        if not scenario_folder.is_dir():
            continue
        
        # Find the latest eval CSV file
        eval_files = sorted(scenario_folder.glob("eval_*.csv"))
        if not eval_files:
            continue
        
        eval_csv = eval_files[-1]
        
        # Load metrics from CSV
        metrics = defaultdict(list)
        sample_count = 0
        
        try:
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_count += 1
                    for key in ["dice_mean", "iou_mean", "f1_mean", "acc_mean", "auprc"]:
                        if key in row:
                            metrics[key].append(float(row[key]))
        except Exception as e:
            print(f"[warn] Failed to read {eval_csv}: {e}")
            continue
        
        # Compute statistics
        if sample_count > 0:
            stats = {}
            for key, values in metrics.items():
                stats[f"{key}_mean"] = statistics.mean(values)
                stats[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
            
            results[scenario_folder.name] = {
                "stats": stats,
                "sample_count": sample_count,
                "eval_file": eval_csv.name,
            }
    
    return results


def print_report(results: dict, out_file: Path = None):
    """Print and optionally save evaluation report."""
    
    lines = []
    
    # Header
    lines.append("=" * 120)
    lines.append("PSO EVALUATION REPORT")
    lines.append("=" * 120)
    lines.append("")
    
    # Summary table
    lines.append("SCENARIO PERFORMANCE SUMMARY")
    lines.append("-" * 120)
    lines.append(
        f"{'Scenario':<25} {'Samples':<10} {'Dice':<18} {'IoU':<18} {'F1':<18} {'Accuracy':<18} {'AUPRC':<18}"
    )
    lines.append("-" * 120)
    
    all_auprc = []
    
    for scenario_name in sorted(results.keys()):
        data = results[scenario_name]
        stats = data["stats"]
        sample_count = data["sample_count"]
        
        dice = stats.get("dice_mean_mean", 0.0)
        iou = stats.get("iou_mean_mean", 0.0)
        f1 = stats.get("f1_mean_mean", 0.0)
        acc = stats.get("acc_mean_mean", 0.0)
        auprc = stats.get("auprc_mean", 0.0)
        
        all_auprc.append(auprc)
        
        lines.append(
            f"{scenario_name:<25} {sample_count:<10} {dice:>6.4f}±{stats.get('dice_mean_std', 0):>5.4f}  "
            f"{iou:>6.4f}±{stats.get('iou_mean_std', 0):>5.4f}  "
            f"{f1:>6.4f}±{stats.get('f1_mean_std', 0):>5.4f}  "
            f"{acc:>6.4f}±{stats.get('acc_mean_std', 0):>5.4f}  "
            f"{auprc:>6.4f}±{stats.get('auprc_std', 0):>5.4f}"
        )
    
    lines.append("-" * 120)
    
    # Overall statistics
    if all_auprc:
        overall_auprc = statistics.mean(all_auprc)
        lines.append(f"{'OVERALL':<25} {len(results):<10} {'':36} {'':18} {'':18} {'AUPRC Mean':<18} {overall_auprc:>6.4f}")
    
    lines.append("=" * 120)
    lines.append("")
    
    # Detailed metrics per scenario
    lines.append("DETAILED METRICS PER SCENARIO")
    lines.append("=" * 120)
    
    for scenario_name in sorted(results.keys()):
        data = results[scenario_name]
        stats = data["stats"]
        sample_count = data["sample_count"]
        eval_file = data["eval_file"]
        
        lines.append("")
        lines.append(f"Scenario: {scenario_name}")
        lines.append(f"  Eval File: {eval_file}")
        lines.append(f"  Samples: {sample_count}")
        lines.append("")
        
        lines.append("  Dice Coefficient:")
        lines.append(f"    Mean: {stats.get('dice_mean_mean', 0):.4f}")
        lines.append(f"    Std:  {stats.get('dice_mean_std', 0):.4f}")
        lines.append(f"    Min:  {stats.get('dice_mean_min', 0):.4f}")
        lines.append(f"    Max:  {stats.get('dice_mean_max', 0):.4f}")
        
        lines.append("  IoU Score:")
        lines.append(f"    Mean: {stats.get('iou_mean_mean', 0):.4f}")
        lines.append(f"    Std:  {stats.get('iou_mean_std', 0):.4f}")
        lines.append(f"    Min:  {stats.get('iou_mean_min', 0):.4f}")
        lines.append(f"    Max:  {stats.get('iou_mean_max', 0):.4f}")
        
        lines.append("  F1 Score:")
        lines.append(f"    Mean: {stats.get('f1_mean_mean', 0):.4f}")
        lines.append(f"    Std:  {stats.get('f1_mean_std', 0):.4f}")
        lines.append(f"    Min:  {stats.get('f1_mean_min', 0):.4f}")
        lines.append(f"    Max:  {stats.get('f1_mean_max', 0):.4f}")
        
        lines.append("  Accuracy:")
        lines.append(f"    Mean: {stats.get('acc_mean_mean', 0):.4f}")
        lines.append(f"    Std:  {stats.get('acc_mean_std', 0):.4f}")
        lines.append(f"    Min:  {stats.get('acc_mean_min', 0):.4f}")
        lines.append(f"    Max:  {stats.get('acc_mean_max', 0):.4f}")
        
        lines.append("  AUPRC:")
        lines.append(f"    Mean: {stats.get('auprc_mean', 0):.4f}")
        lines.append(f"    Std:  {stats.get('auprc_std', 0):.4f}")
        lines.append(f"    Min:  {stats.get('auprc_min', 0):.4f}")
        lines.append(f"    Max:  {stats.get('auprc_max', 0):.4f}")
    
    lines.append("")
    lines.append("=" * 120)
    
    # Print to console
    report_text = "\n".join(lines)
    print(report_text)
    
    # Save to file if requested
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport saved to: {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Generate PSO evaluation report from scenario results")
    ap.add_argument(
        "--eval-dir",
        default="./output/pso_eval",
        help="Directory containing scenario evaluation folders",
    )
    ap.add_argument(
        "--out-file",
        default=None,
        help="Optional: save report to file (e.g., ./output/pso_eval_report.txt)",
    )
    args = ap.parse_args()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"[error] Evaluation directory not found: {eval_dir}")
        return
    
    results = load_scenario_results(eval_dir)
    
    if not results:
        print(f"[error] No evaluation results found in {eval_dir}")
        return
    
    out_file = Path(args.out_file) if args.out_file else None
    print_report(results, out_file)


if __name__ == "__main__":
    main()
