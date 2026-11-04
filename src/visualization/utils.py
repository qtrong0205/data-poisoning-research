import torch
import random
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from torch import nn


def plot_transformed_images(dataloader: torch.utils.data.DataLoader,
                             class_names,
                             n: int = 3,
                             seed: int = None):
    if seed:
        random.seed(seed)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images, labels = next(iter(dataloader))

    batch_size = images.shape[0]
    n = min(n, batch_size) 
    indices = random.sample(range(batch_size), n)

    for idx in indices:
        img = images[idx]
        label = labels[idx]

        img_unnormalized = img * std + mean
        img_unnormalized = img_unnormalized.permute(1, 2, 0).clamp(0, 1)

        plt.figure(figsize=(12, 7))
        plt.imshow(img_unnormalized)
        plt.title(class_names[label])
        plt.axis("off")

def plot_loss_curves(model_results):
    """Plots training curves of a results dictionary."""
    epochs = range(len(model_results["train_loss"]))

    train_loss = model_results["train_loss"]
    train_acc = model_results["train_acc"]
    test_loss = model_results["test_loss"]
    test_acc = model_results["test_acc"]

    plt.figure(figsize=(15, 7))

    # Plotting loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plotting accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, test_acc, label="Test acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def pred_and_plot_images(model: nn.Module,
                         dataset: torch.utils.data.Dataset,
                         class_names: List[str],
                         device: torch.device,
                         n: int = 5,
                         seed: Optional[int] = None) -> None:
    """Predicts on random images from a dataset and plots them with
    predicted vs. true class labels.
    """
    if seed is not None:
        random.seed(seed)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)

    random_indices = random.sample(range(len(dataset)), k=n)

    # Making prediction
    model.eval()
    with torch.inference_mode():
        for i in random_indices:
            X, y = dataset[i]

            X = X.to(device)

            y_pred = model(X.unsqueeze(0)) 

            pred_class_idx = torch.argmax(y_pred, dim=1).item()
            probs = torch.softmax(y_pred, dim=1)

            pred_prob = probs.max(dim=1).values.item()  
            pred_class = class_names[pred_class_idx]
            true_class = class_names[y]

            title = (
                f"True: {true_class}\n"
                f"Pred: {pred_class} ({pred_prob:.2%})"
            )

            img_unnormalized = X * std + mean
            img_unnormalized = img_unnormalized.permute(1, 2, 0).clamp(0, 1).cpu()

            # Plotting predictions
            plt.figure(figsize=(12, 7))
            plt.imshow(img_unnormalized)
            plt.title(title, color="green" if true_class == pred_class else "red")
            plt.axis("off")

def export_to_onnx(model: nn.Module,
                   example_input: torch.Tensor,
                   save_path: str,
                   input_names: Optional[List[str]] = None,
                   output_names: Optional[List[str]] = None,
                   dynamo: bool = True) -> None:
    """Exports a trained PyTorch model to ONNX format."""
    model.eval()

    torch.onnx.export(
        model=model,
        args=(example_input,),
        f=save_path,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamo=dynamo,
    )

    print(f"Model exported to {save_path}")