"""Evaluation helpers for Label Flipping attacks."""

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.evaluation.baseline.eval import (
    calculate_metrics,
    make_predictions,
    plot_confusion_matrix,
)


def calculate_per_class_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """
    Calculate precision, recall, F1 and support for each class.
    """

    if preds.ndim != 1 or labels.ndim != 1:
        raise ValueError("preds and labels must be one-dimensional.")

    if preds.numel() != labels.numel():
        raise ValueError("preds and labels must have the same length.")

    if labels.numel() == 0:
        raise ValueError("Cannot calculate per-class metrics on empty data.")

    if num_classes is None:
        num_classes = len(class_names) if class_names else (
            int(max(labels.max(), preds.max()).item()) + 1
        )

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes.")

    metric_labels = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels.numpy(),
        preds.numpy(),
        labels=metric_labels,
        average=None,
        zero_division=0,
    )

    results = {}

    for index in range(num_classes):
        class_name = (
            class_names[index] if class_names is not None else str(index)
        )

        results[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "support": int(support[index]),
        }

    return results


def calculate_flipped_label_metrics(
    preds: torch.Tensor,
    poisoned_labels: torch.Tensor,
    sample_ids: Sequence[Any],
    flipped_sample_ids: Sequence[Any],
) -> dict[str, Any]:
    """
    Measure how often the model predicts the corrupted label
    for samples whose training labels were flipped.

    poisoned_labels are the labels returned by the poisoned
    training dataloader, not the original labels.
    """

    if preds.ndim != 1 or poisoned_labels.ndim != 1:
        raise ValueError("preds and poisoned_labels must be one-dimensional.")

    if preds.numel() != poisoned_labels.numel():
        raise ValueError("preds and poisoned_labels must have the same length.")

    if len(sample_ids) != preds.numel():
        raise ValueError("sample_ids length must match predictions.")

    if not flipped_sample_ids:
        raise ValueError("flipped_sample_ids must not be empty.")

    flipped_ids = set(flipped_sample_ids)

    selected_indices = [
        index
        for index, sample_id in enumerate(sample_ids)
        if sample_id in flipped_ids
    ]

    if not selected_indices:
        raise ValueError(
            "No flipped samples were found in the training predictions."
        )

    indices = torch.tensor(selected_indices, dtype=torch.long)

    selected_preds = preds[indices]
    selected_labels = poisoned_labels[indices]

    predicted_as_flipped_label = selected_preds == selected_labels

    count = len(selected_indices)
    matched_count = int(predicted_as_flipped_label.sum().item())

    return {
        "num_flipped_samples": count,
        "predicted_as_flipped_label_count": matched_count,
        "predicted_as_flipped_label_rate": float(matched_count / count),
    }


def _save_predictions(
    predictions_path: str | Path,
    sample_ids: Sequence[Any],
    labels: torch.Tensor,
    preds: torch.Tensor,
) -> Path:
    """Save clean test predictions to CSV."""

    if len(sample_ids) != labels.numel():
        sample_ids = [
            str(index) for index in range(labels.numel())
        ]

    path = Path(predictions_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["sample_id", "y_true", "y_pred"])

        for index in range(labels.numel()):
            writer.writerow([
                sample_ids[index],
                int(labels[index]),
                int(preds[index]),
            ])

    return path


def evaluate_model(
    model: torch.nn.Module,
    clean_dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str] | None = None,
    baseline_metrics: Mapping[str, float] | None = None,
    poisoned_train_dataloader: DataLoader | None = None,
    flipped_sample_ids: Sequence[Any] | None = None,
    predictions_path: str | Path | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a Label Flipping model.

    Main evaluation:
        - Clean Accuracy, Macro Precision, Macro Recall, Macro-F1
        - Delta Accuracy and Macro-F1 against Baseline (optional)
        - Per-class Precision, Recall and F1
        - Confusion Matrix

    Optional evaluation:
        - Percentage of flipped training samples predicted
          according to their corrupted labels.

    The clean test set must retain its original labels.
    """

    if model is None:
        raise ValueError("model must be provided.")

    if clean_dataloader is None:
        raise ValueError("clean_dataloader must be provided.")

    if len(clean_dataloader) == 0:
        raise ValueError("Cannot evaluate an empty clean_dataloader.")

    if (poisoned_train_dataloader is None) != (flipped_sample_ids is None):
        raise ValueError(
            "poisoned_train_dataloader and flipped_sample_ids "
            "must be provided together."
        )

    num_classes = len(class_names) if class_names is not None else None


    clean_preds, clean_labels, clean_logits, clean_ids = make_predictions(
        model,
        clean_dataloader,
        device,
        collect_details=True,
    )

    clean_metrics = calculate_metrics(
        clean_preds,
        clean_labels,
        num_classes=num_classes,
    )

    per_class_metrics = calculate_per_class_metrics(
        clean_preds,
        clean_labels,
        class_names=class_names,
        num_classes=num_classes,
    )

    clean_figure, clean_axes = plot_confusion_matrix(
        clean_preds,
        clean_labels,
        class_names=class_names,
        num_classes=num_classes,
        title="Label Flipping Clean Test Confusion Matrix",
    )


    if baseline_metrics is not None:
        required_metrics = {"accuracy", "f1_score"}

        missing_metrics = required_metrics - baseline_metrics.keys()

        if missing_metrics:
            raise ValueError(
                f"baseline_metrics is missing: {sorted(missing_metrics)}"
            )

        clean_metrics["delta_accuracy"] = float(
            baseline_metrics["accuracy"] - clean_metrics["accuracy"]
        )

        clean_metrics["delta_macro_f1"] = float(
            baseline_metrics["f1_score"] - clean_metrics["f1_score"]
        )

    clean_result: dict[str, Any] = {
        "metrics": clean_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": clean_figure,
        "confusion_matrix_axes": clean_axes,
    }

    if predictions_path is not None:
        saved_path = _save_predictions(
            predictions_path,
            clean_ids,
            clean_labels,
            clean_preds,
        )
        clean_result["predictions_path"] = str(saved_path)

    if return_details:
        clean_result["predictions"] = {
            "sample_id": clean_ids,
            "y_true": clean_labels.tolist(),
            "y_pred": clean_preds.tolist(),
            "logits": clean_logits.tolist(),
        }

        clean_result["confusion_matrix_values"] = confusion_matrix(
            clean_labels.numpy(),
            clean_preds.numpy(),
            labels=list(range(num_classes))
            if num_classes is not None
            else None,
        ).tolist()


    attack_result: dict[str, Any] = {}

    if poisoned_train_dataloader is not None:
        if len(poisoned_train_dataloader) == 0:
            raise ValueError("Cannot evaluate an empty poisoned_train_dataloader.")

        train_preds, poisoned_labels, _, train_ids = make_predictions(
            model,
            poisoned_train_dataloader,
            device,
            collect_details=True,
        )

        flipped_metrics = calculate_flipped_label_metrics(
            preds=train_preds,
            poisoned_labels=poisoned_labels,
            sample_ids=train_ids,
            flipped_sample_ids=flipped_sample_ids,
        )

        attack_result["flipped_label_adoption"] = flipped_metrics

    result = {
        "clean": clean_result,
    }

    if attack_result:
        result["attack"] = attack_result

    return result