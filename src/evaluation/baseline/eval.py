import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def make_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
):
    """
    Generate predictions and collect ground-truth labels.
    """

    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            # Move results to CPU to avoid accumulating tensors on GPU
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return all_preds, all_labels


def calculate_acc(
    preds: torch.Tensor,
    labels: torch.Tensor
):
    """
    Calculate classification accuracy.
    """

    acc = accuracy_score(
        labels.numpy(),
        preds.numpy()
    )

    print(f"Accuracy of model: {acc:.4f}")

    return acc


def plot_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names=None
):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(
        labels.numpy(),
        preds.numpy()
    )

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
