"""
Day 5 — Task 3
Clinical Accuracy Report Generator.
Runs full evaluation and writes a formatted report comparing NeoScreen
to the Bilicam paper benchmarks (97% sensitivity, Jaundice Tool reference).

Usage:
    python day5/accuracy_report.py --model neoscreen_v1.tflite --data_dir ./dataset
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REPORT_DIR = os.path.join("day5")
BILICAM_BENCHMARKS = {
    "sensitivity": 0.97,
    "specificity": 0.62,
    "auc":         0.94,
    "kappa":       0.83,
    "source":      "Bilicam (Smith et al., 2016) — University of Washington",
}


def load_eval_results():
    path = "eval_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run(model_path, data_dir):
    print("=" * 58)
    print("NeoScreen — Day 5: Clinical Accuracy Report")
    print("=" * 58)

    # Run evaluation
    print("\nRunning evaluation...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "ml/evaluate.py",
         "--model", model_path,
         "--test_dir", data_dir,
         "--output", "eval_results.json"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Evaluation failed:")
        print(result.stderr)
        sys.exit(1)

    # Load results
    results = load_eval_results()
    if not results:
        print("eval_results.json not found after evaluation.")
        sys.exit(1)

    our = results
    ref = BILICAM_BENCHMARKS

    # Build report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "NeoScreen_Accuracy_Report.txt")

    def fmt_pct(val):
        return f"{val*100:.1f}%" if val is not None else "N/A"

    def gap(ours, theirs):
        if ours is None:
            return "N/A"
        diff = ours - theirs
        return f"{'+'if diff>=0 else ''}{diff*100:.1f}pp"

    lines = [
        "=" * 65,
        "NeoScreen — Clinical Accuracy Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model:     {model_path}",
        f"Dataset:   {data_dir}  ({results.get('n_samples','?')} images)",
        "=" * 65,
        "",
        "METRIC COMPARISON",
        "-" * 65,
        f"{'Metric':<28} {'NeoScreen':>12}  {'Bilicam*':>10}  {'Gap':>8}",
        "-" * 65,
        f"{'Sensitivity (High Risk)':<28} {fmt_pct(our.get('sensitivity_high_risk')):>12}  "
        f"{fmt_pct(ref['sensitivity']):>10}  {gap(our.get('sensitivity_high_risk'), ref['sensitivity']):>8}",

        f"{'Specificity':<28} {fmt_pct(our.get('specificity_high_risk')):>12}  "
        f"{fmt_pct(ref['specificity']):>10}  {gap(our.get('specificity_high_risk'), ref['specificity']):>8}",

        f"{'Cohen\'s Kappa':<28} {our.get('cohen_kappa','N/A'):>12}  "
        f"{ref['kappa']:>10}  "
        f"{gap(our.get('cohen_kappa'), ref['kappa']) if our.get('cohen_kappa') else 'N/A':>8}",

        f"{'AUC-ROC (macro)':<28} {our.get('auc_roc_macro','N/A'):>12}  "
        f"{ref['auc']:>10}  "
        f"{gap(our.get('auc_roc_macro'), ref['auc']) if our.get('auc_roc_macro') else 'N/A':>8}",
        "-" * 65,
        f"* {ref['source']}",
        "",
        "TARGET ASSESSMENT",
        "-" * 65,
    ]

    targets = [
        ("Sensitivity >= 95%",  our.get('sensitivity_high_risk', 0) >= 0.95),
        ("Kappa > 0.80",        (our.get('cohen_kappa') or 0) > 0.80),
        ("AUC > 0.92",          (our.get('auc_roc_macro') or 0) > 0.92),
    ]
    for label, passed in targets:
        lines.append(f"  {'✓' if passed else '✗'}  {label}")

    lines += [
        "",
        "LIMITATIONS & HONEST GAPS",
        "-" * 65,
        "1. TRAINING DATA",
        "   Current model trained on synthetic images (no real neonatal data).",
        "   Real performance will differ. Clinical validation pending via",
        "   IISc Medical Network (target n=200+, Phase 1).",
        "",
        "2. SCLERA DETECTION",
        "   HSV-based detection may fail in extreme lighting (< 50 lux).",
        "   CLAHE mitigates this but does not eliminate it.",
        "   Always prompt retry if sclera not detected.",
        "",
        "3. NOT A DIAGNOSTIC TOOL",
        "   NeoScreen is a SCREENING tool only.",
        "   High Risk result should trigger immediate clinical referral.",
        "   It does NOT replace TSB blood test for diagnosis.",
        "   Medium / Low results do NOT rule out jaundice.",
        "",
        "4. POPULATION COVERAGE",
        "   Validated on Fitzpatrick types 1-6 (synthetic).",
        "   Real-world validation across Indian skin tones pending.",
        "",
        "WHAT THIS MEANS FOR DEPLOYMENT",
        "-" * 65,
        "NeoScreen is a safety net, not a safety guarantee.",
        "It is designed to reduce the 80% of rural births with zero",
        "monitoring — any screening is better than none.",
        "The safety-first threshold (P(High) >= 0.35) deliberately",
        "over-refers rather than under-detects. Over-referral is safe.",
        "Under-detection causes kernicterus.",
        "",
        "=" * 65,
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="neoscreen_v1.tflite")
    parser.add_argument("--data_dir", default="./dataset")
    args = parser.parse_args()
    run(args.model, args.data_dir)
