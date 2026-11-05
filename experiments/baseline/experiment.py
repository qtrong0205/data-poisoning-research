import json
import random
import yaml
import torch

from pathlib import Path

from src.data.splitting.split_dataset import split_dataset

from src.data.loading.data_setup import (
    ImageDataset,
    create_transform,
    create_dataloader
)

from src.models.baseline.model_builder import build_model
from src.models.baseline.engine import train

from src.evaluation.baseline.eval import evaluate_model


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline" / "baseline.yaml"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "baseline"

RESULT_DIR = PROJECT_ROOT / "experiments" / "baseline" / "results"

LOG_DIR = PROJECT_ROOT / "experiments" / "baseline" / "logs"


# ==================================================
# 2. CONFIGURATION
# ==================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================
# 3. DATA PREPARATION
# ==================================================

def prepare_dataset(config):
    dataset_config = config["dataset"]

    raw_dir = PROJECT_ROOT / dataset_config["data_dir"]
    split_dir = PROJECT_ROOT / dataset_config["split_dir"]

    print("\n[1/5] Splitting dataset...")

    split_dataset(
        source_dir=str(raw_dir),
        output_dir=str(split_dir),
        train_ratio=dataset_config["train_ratio"],
        random_state=config["experiment"]["seed"]
    )

    print("\n[2/5] Creating datasets and dataloaders...")

    image_size = tuple(dataset_config["image_size"])

    train_transform = create_transform(
        size=image_size,
        train=True
    )

    test_transform = create_transform(
        size=image_size,
        train=False
    )

    train_dataset = ImageDataset(
        targ_dir=str(split_dir / "train"),
        transform=train_transform,
        subset_ratio=dataset_config["subset_ratio"],
        seed=config["experiment"]["seed"]
    )

    test_dataset = ImageDataset(
        targ_dir=str(split_dir / "test"),
        transform=test_transform
    )

    # Ensure consistent class mapping
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings do not match.")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")

    return train_loader, test_loader, train_dataset.classes


# ==================================================
# 4. MODEL TRAINING
# ==================================================

def train_baseline(config, train_loader, device):
    print("\n[3/5] Building model...")

    model = build_model(
        num_classes=config["dataset"]["num_classes"],
        device=device,
        pretrained=config["model"]["pretrained"]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    print("\n[4/5] Training baseline model...")

    training_results = train(
        model=model,
        train_dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config["training"]["epochs"],
        device=device,
        checkpoint_path=str(checkpoint_path)
    )

    return model, training_results, checkpoint_path


# ==================================================
# 5. EVALUATION
# ==================================================

def evaluate_baseline(model, test_loader, device, class_names):
    print("\n[5/5] Evaluating baseline model...")

    accuracy, fig, ax = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device
    )

    if fig is not None:
        fig.savefig(
            RESULT_DIR / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        import matplotlib.pyplot as plt
        plt.close(fig)

    return {
        "accuracy": float(accuracy)
    }


# ==================================================
# 6. MAIN EXPERIMENT
# ==================================================

def run_experiment():
    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    set_seed(config["experiment"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("BASELINE EXPERIMENT")
    print("=" * 60)

    print(f"Device: {device}")

    # Prepare data
    train_loader, test_loader, class_names = prepare_dataset(config)

    # Build and train
    model, training_results, checkpoint_path = train_baseline(
        config=config,
        train_loader=train_loader,
        device=device
    )

    # Evaluate
    evaluation_results = evaluate_baseline(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    # Save metrics
    results = {
        "experiment": config["experiment"]["name"],
        "seed": config["experiment"]["seed"],
        "dataset": config["dataset"]["name"],
        "num_classes": len(class_names),
        "num_train_samples": len(train_loader.dataset),
        "num_test_samples": len(test_loader.dataset),
        "training": training_results,
        "evaluation": evaluation_results,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT))
    }

    results_path = RESULT_DIR / "metrics.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 60)
    print("BASELINE EXPERIMENT COMPLETED")
    print("=" * 60)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {results_path}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    run_experiment()
