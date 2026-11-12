import torch
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def make_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    collect_details: bool = False,
):
    """
    Generate predictions and collect ground-truth labels.
    """

    model.eval()

    total_batches = len(dataloader)
    if total_batches == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")
    all_preds = []
    all_labels = []
    all_logits = []
    all_sample_ids = []

    with torch.inference_mode():
        for batch in dataloader:
            X, y = batch[:2]
            X = X.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            # Move results to CPU to avoid accumulating tensors on GPU
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            if collect_details:
                all_logits.append(logits.cpu())
                all_sample_ids.extend(batch[2] if len(batch) > 2 else [])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if not collect_details:
        return all_preds, all_labels
    return all_preds, all_labels, torch.cat(all_logits), all_sample_ids


def calculate_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int | None = None,
):
    """
    Calculate classification metrics using macro averaging.
    """

    metric_labels = list(range(num_classes)) if num_classes is not None else None
    y_true = labels.numpy()
    y_pred = preds.numpy()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
    }

    return metrics


def plot_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names=None,
    num_classes=None,
):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=(
        list(range(num_classes)) if num_classes is not None else None
    ))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(16, 16))

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Baseline Confusion Matrix")
    plt.tight_layout()

    return fig, ax


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   class_names=None,
                   predictions_path: str | Path | None = None,
                   return_details: bool = False):
    print("Evaluation step 1/4: generating predictions.", flush=True)
    preds, labels, logits, sample_ids = make_predictions(
        model, dataloader, device, collect_details=True
    )

    print("Evaluation step 2/4: calculating metrics.", flush=True)
    metrics = calculate_metrics(
        preds,
        labels,
        num_classes=len(class_names) if class_names else None,
    )
    acc = metrics["accuracy"]

    print("Evaluation step 3/4: plotting confusion matrix.", flush=True)
    fig, ax = plot_confusion_matrix(
        preds, labels, class_names=class_names, num_classes=len(class_names)
        if class_names else None
    )
    if predictions_path is not None:
        path = Path(predictions_path)
        print(f"Evaluation step 4/4: writing predictions to {path.resolve()}.", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["sample_id", "y_true", "y_pred", "logits"])
            for index in range(len(labels)):
                writer.writerow([
                    sample_ids[index] if index < len(sample_ids) else str(index),
                    int(labels[index]),
                    int(preds[index]),
                    logits[index].tolist(),
                ])

    details = {
        "y_true": labels.tolist(),
        "y_pred": preds.tolist(),
        "logits": logits.tolist(),
        "sample_id": sample_ids,
        "metrics": metrics,
    }
    return (acc, fig, ax, details) if return_details else (acc, fig, ax)