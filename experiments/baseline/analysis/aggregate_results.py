
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "experiments/baseline/results"
OUTPUT_DIR = PROJECT_ROOT / "experiments/baseline/analysis/outputs"

SEEDS = [42, 43, 44]
MODE = "confirm"

METRICS = ["accuracy", "precision", "recall", "f1_score"]


def load_metrics(seed):
    path = RESULTS_DIR / MODE / f"seed_{seed}" / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["evaluation"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in SEEDS:
        evaluation = load_metrics(seed)

        row = {"seed": seed}

        for metric in METRICS:
            row[metric] = float(evaluation[metric])

        rows.append(row)

    df = pd.DataFrame(rows)

    summary = pd.DataFrame({
        "metric": METRICS,
        "mean": [df[m].mean() for m in METRICS],
        "std": [df[m].std(ddof=1) for m in METRICS],
    })

    summary["mean_std"] = summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1
    )

    df.to_csv(
        OUTPUT_DIR / "baseline_seed_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "baseline_summary.csv",
        index=False
    )

    print("\n=== RESULTS PER SEED ===")
    print(df.to_string(index=False))

    print("\n=== STATISTICAL SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
