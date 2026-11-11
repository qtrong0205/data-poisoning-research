import torch
from torch import nn
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from timeit import default_timer as timer
from pathlib import Path

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target model to training mode, then runs through all of
    the required training steps (forward pass, loss calculation,
    optimizer step) for one epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics, in the
        form (train_loss, train_acc).
    """
    model.train()

    train_loss, train_acc = 0, 0

    sample_count = 0
    for _, batch in enumerate(dataloader):
        X, y = batch[:2]
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_pred, y)
        batch_size = len(y)
        sample_count += batch_size
        train_loss += loss.item() * batch_size

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step (gradient descent)
        optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch
    if sample_count == 0:
        raise ValueError("Cannot train on an empty dataloader.")
    train_loss /= sample_count
    train_acc /= sample_count
    return train_loss, train_acc

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target model to "eval" mode, then performs a forward pass
    on a testing dataset without computing gradients.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics, in the
        form (test_loss, test_acc).
    """
    model.eval()

    test_loss, test_acc = 0, 0
    sample_count = 0

    with torch.inference_mode():
        for _, batch in enumerate(dataloader):
            X, y = batch[:2]
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y)
            batch_size = len(y)
            sample_count += batch_size
            test_loss += loss.item() * batch_size

            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += (test_pred_class == y).sum().item()

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")
    test_loss /= sample_count
    test_acc /= sample_count
    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    checkpoint_path: str = "best_model.pth",
) -> Dict[str, List[float]]:
    """
    Train model using training dataset only.
    Save the final model checkpoint.
    """

    model_results = {
        "train_loss": [],
        "train_acc": [],
        "total_time": 0
    }

    model.to(device)

    start_time = timer()

    epoch_progress = tqdm(
        range(epochs),
        desc="Training epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f}",
            flush=True,
        )
        epoch_progress.set_postfix(
            loss=f"{train_loss:.4f}",
            accuracy=f"{train_acc:.4f}",
        )

        model_results["train_loss"].append(train_loss)
        model_results["train_acc"].append(train_acc)

    total_time = timer() - start_time
    model_results["total_time"] = total_time

    # Save final checkpoint
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), checkpoint_path)

    print(f"\nCheckpoint saved: {checkpoint_path}", flush=True)
    print(f"Total training time: {total_time:.2f} seconds", flush=True)

    return model_results
