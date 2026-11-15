"""Evaluation helpers for all-to-one backdoor models."""

import csv
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from src.evaluation.baseline.eval import (
    calculate_metrics,
    make_predictions,
    plot_confusion_matrix,
)


def _validate_target_label(target_label: int) -> None:
    if not isinstance(target_label, int) or isinstance(target_label, bool):
        raise TypeError("target_label must be an integer.")
    if target_label < 0:
        raise ValueError("target_label must be non-negative.")


def _validate_asr_inputs(
    preds: torch.Tensor,
    labels: torch.Tensor,
    target_label: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_target_label(target_label)
    if not isinstance(preds, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("preds and labels must be torch.Tensor instances.")
    if preds.ndim != 1 or labels.ndim != 1:
        raise ValueError("preds and labels must be one-dimensional.")
    if preds.numel() != labels.numel():
        raise ValueError("preds and labels must have the same number of elements.")
    if labels.numel() == 0:
        raise ValueError("Cannot calculate ASR on an empty dataset.")

    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    if torch.any(labels == target_label).item():
        raise ValueError(
            "ASR evaluation labels must exclude the target class."
        )
    return preds, labels


def calculate_asr(
    preds: torch.Tensor,
    labels: torch.Tensor,
    target_label: int,
) -> float:
    """Return the fraction of triggered samples predicted as the target class."""
    preds, labels = _validate_asr_inputs(preds, labels, target_label)
    return float((preds == target_label).sum().item() / labels.numel())


def calculate_per_class_asr(
    preds: torch.Tensor,
    labels: torch.Tensor,
    target_label: int,
    class_names: Sequence[str] | None = None,
) -> dict[int | str, float]:
    """Return ASR grouped by each non-target source class present in the data."""
    preds, labels = _validate_asr_inputs(preds, labels, target_label)

    per_class_asr: dict[int | str, float] = {}
    for label in torch.unique(labels, sorted=True).tolist():
        if class_names is not None:
            if label < 0 or label >= len(class_names):
                raise ValueError(
                    f"Label {label} has no corresponding class name."
                )
            class_key: int | str = class_names[label]
        else:
            class_key = label

        class_mask = labels == label
        per_class_asr[class_key] = float(
            ((preds == target_label) & class_mask).sum().item()
            / class_mask.sum().item()
        )
    return per_class_asr


def _prediction_details(
    preds: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[Any],
    target_label: int,
) -> dict[str, list[Any]]:
    count = labels.numel()
    if not sample_ids:
        sample_ids = [str(index) for index in range(count)]
    elif len(sample_ids) != count:
        raise ValueError("The number of sample IDs does not match predictions.")
    return {
        "sample_id": sample_ids,
        "y_true": labels.tolist(),
        "y_pred": preds.tolist(),
        "is_target_prediction": (preds == target_label).tolist(),
    }


def _save_predictions(
    predictions_path: str | Path,
    details: dict[str, list[Any]],
    target_label: int,
) -> Path:
    path = Path(predictions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sample_id", "y_true", "y_pred", "is_target_prediction", "target_label"]
        )
        for sample_id, y_true, y_pred, is_target_prediction in zip(
            details["sample_id"],
            details["y_true"],
            details["y_pred"],
            details["is_target_prediction"],
        ):
            writer.writerow(
                [sample_id, y_true, y_pred, is_target_prediction, target_label]
            )
    return path


def _predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loader_name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    if dataloader is None:
        raise ValueError(f"{loader_name} must be provided.")
    if len(dataloader) == 0:
        raise ValueError(f"Cannot evaluate an empty {loader_name}.")

    preds, labels, _, sample_ids = make_predictions(
        model, dataloader, device, collect_details=True
    )
    if labels.numel() == 0:
        raise ValueError(f"Cannot evaluate an empty {loader_name}.")
    return preds, labels, sample_ids


def evaluate_model(
    model: torch.nn.Module,
    clean_dataloader: DataLoader,
    asr_dataloader: DataLoader,
    device: torch.device,
    target_label: int,
    class_names: Sequence[str] | None = None,
    clean_predictions_path: str | Path | None = None,
    asr_predictions_path: str | Path | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """Evaluate clean classification quality and triggered attack success."""
    _validate_target_label(target_label)
    if model is None:
        raise ValueError("model must be provided.")
    if class_names is not None and target_label >= len(class_names):
        raise ValueError("target_label has no corresponding class name.")

    clean_preds, clean_labels, clean_ids = _predict(
        model, clean_dataloader, device, "clean_dataloader"
    )
    clean_metrics = calculate_metrics(
        clean_preds,
        clean_labels,
        num_classes=len(class_names) if class_names else None,
    )
    clean_figure, clean_axes = plot_confusion_matrix(
        clean_preds,
        clean_labels,
        class_names=class_names,
        num_classes=len(class_names) if class_names else None,
        title="Backdoor Clean Evaluation Confusion Matrix",
    )
    clean_details = _prediction_details(
        clean_preds, clean_labels, clean_ids, target_label
    )

    asr_preds, asr_labels, asr_ids = _predict(
        model, asr_dataloader, device, "asr_dataloader"
    )
    asr = calculate_asr(asr_preds, asr_labels, target_label)
    per_class_asr = calculate_per_class_asr(
        asr_preds, asr_labels, target_label, class_names
    )
    asr_details = _prediction_details(asr_preds, asr_labels, asr_ids, target_label)

    clean_result: dict[str, Any] = {
        "metrics": clean_metrics,
        "confusion_matrix": clean_figure,
        "confusion_matrix_axes": clean_axes,
    }
    backdoor_result: dict[str, Any] = {
        "metrics": {"asr": asr},
        "per_class_asr": per_class_asr,
        "target_label": target_label,
    }

    if clean_predictions_path is not None:
        clean_result["predictions_path"] = str(
            _save_predictions(clean_predictions_path, clean_details, target_label)
        )
    if asr_predictions_path is not None:
        backdoor_result["predictions_path"] = str(
            _save_predictions(asr_predictions_path, asr_details, target_label)
        )
    if return_details:
        clean_result["predictions"] = clean_details
        backdoor_result["predictions"] = asr_details

    return {"clean": clean_result, "backdoor": backdoor_result}
