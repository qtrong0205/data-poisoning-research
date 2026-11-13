
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "experiments/baseline/results"
SPLIT_DIR = PROJECT_ROOT / "data/splits"

OUTPUT_DIR = PROJECT_ROOT / "experiments/baseline/analysis/outputs"

MODE = "confirm"
SEEDS = [42, 43, 44]


def load_class_mapping():
    path = SPLIT_DIR / "class_to_idx.json"

    with open(path, "r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    return {
        int(index): name
        for name, index in class_to_idx.items()
    }


def load_predictions(seed):
    path = RESULTS_DIR / MODE / f"seed_{seed}" / "predictions.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"y_true", "y_pred"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {path}: {required - set(df.columns)}"
        )

    return df


def calculate_per_class(df, idx_to_class):
    class_ids = sorted(idx_to_class.keys())

    report = classification_report(
        df["y_true"],
        df["y_pred"],
        labels=class_ids,
        target_names=[idx_to_class[i] for i in class_ids],
        output_dict=True,
        zero_division=0
    )

    rows = []

    for class_id in class_ids:
        name = idx_to_class[class_id]
        metrics = report[name]

        rows.append({
            "class_id": class_id,
            "class_name": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"],
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idx_to_class = load_class_mapping()
    all_results = []

    for seed in SEEDS:
        predictions = load_predictions(seed)

        per_class = calculate_per_class(
            predictions,
            idx_to_class
        )

        per_class["seed"] = seed

        per_class.to_csv(
            OUTPUT_DIR / f"per_class_seed_{seed}.csv",
            index=False
        )

        all_results.append(per_class)

        print(f"Processed seed {seed}")

    combined = pd.concat(all_results, ignore_index=True)

    summary = (
        combined
        .groupby(["class_id", "class_name"])
        .agg(
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1_score=("f1_score", "mean"),
            std_f1_score=("f1_score", "std"),
            mean_support=("support", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "per_class_summary.csv",
        index=False
    )

    # Visualize mean F1-score across seeds
    plot_df = summary.sort_values("mean_f1_score")

    plt.figure(figsize=(13, 15))

    plt.barh(
        plot_df["class_name"],
        plot_df["mean_f1_score"],
        xerr=plot_df["std_f1_score"],
        capsize=2
    )

    plt.xlabel("Mean F1-score")
    plt.ylabel("Class")
    plt.title("Baseline Per-class F1-score (Mean ± Std)")
    plt.xlim(0, 1.0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "per_class_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n=== PER-CLASS SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
