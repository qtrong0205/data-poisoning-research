from pathlib import Path
import pandas as pd
import yaml
import json

# Import đúng tên file posioning.py của bạn
from posioning import flip_labels_random, flip_labels_targeted

# Tính đường dẫn gốc của project dựa vào vị trí file hiện tại
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
        
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_and_validate_inputs(config):
    data_config = config["data"]
    attack_config = config["attack"]
    strategy = attack_config.get("strategy", "random")

    train_path = PROJECT_ROOT / data_config["train_manifest"]
    test_path = PROJECT_ROOT / data_config["test_manifest"]
    mapping_path = PROJECT_ROOT / data_config["class_mapping"]

    for path in [train_path, test_path, mapping_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)

    source_label = None
    target_label = None
    
    if strategy == "targeted":
        source_class = attack_config.get("source_class")
        target_class = attack_config.get("target_class")
        
        if not source_class or not target_class:
            raise ValueError("Phải cấu hình source_class và target_class cho chiến lược targeted")
        if source_class not in class_to_idx or target_class not in class_to_idx:
            raise ValueError("Source hoặc Target class không khớp với class_to_idx")
            
        source_label = class_to_idx[source_class]
        target_label = class_to_idx[target_class]

    return train_df, test_df, class_to_idx, source_label, target_label, strategy

def select_scope_samples(train_df, dataset_scope):
    if dataset_scope == "subset":
        if "in_subset" not in train_df.columns:
            raise ValueError("Không tìm thấy cột 'in_subset'")
        return train_df[train_df["in_subset"] == True].copy()
    return train_df.copy()

def save_poisoned_data(poisoned_df, output_dir, metadata):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_train.csv"
    metadata_path = output_dir / "metadata.json"

    poisoned_df.to_csv(manifest_path, index=False, encoding="utf-8")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)
        
    return manifest_path

def main(experiment_id, config_path):
    config = load_config(config_path)
    
    attack_config = config["attack"]
    data_config = config["data"]
    experiment_config = config["experiments"][experiment_id]
    
    poisoning_rates = experiment_config["poisoning_rates"]
    dataset_scope = experiment_config.get("dataset_scope", "full")
    selection_seed = attack_config.get("selection_seed", 42)

    train_df, _, class_to_idx, source_label, target_label, strategy = load_and_validate_inputs(config)
    num_classes = len(class_to_idx)
    
    scoped_df = select_scope_samples(train_df, dataset_scope)
    # Reset index để khớp với mảng trả về của numpy
    scoped_df = scoped_df.reset_index(drop=True)
    labels_array = scoped_df["label"].values

    print(f"\nChuẩn bị dữ liệu cho: {experiment_id} | Chiến lược: {strategy.upper()}")
    
    output_root = Path(data_config["poisoned_manifest_root"]) / experiment_id

    for rate in poisoning_rates:
        # 1. Gọi hàm đổi nhãn
        if strategy == "random":
            new_labels, poisoned_idx = flip_labels_random(labels_array, rate, num_classes, selection_seed)
        else: # targeted
            denominator = attack_config.get("denominator", "source_class")
            new_labels, poisoned_idx = flip_labels_targeted(
                labels_array, rate, source_label, target_label, selection_seed, denominator
            )

        poison_count = len(poisoned_idx)
        
        # 2. Cập nhật Dataframe mới
        poisoned_df = scoped_df.copy()
        poisoned_df["original_label"] = poisoned_df["label"]
        poisoned_df["label"] = new_labels
        poisoned_df["is_poisoned"] = False
        poisoned_df.loc[poisoned_idx, "is_poisoned"] = True

        # 3. Lưu thông tin metadata và file CSV
        metadata = {
            "experiment_id": experiment_id,
            "strategy": strategy,
            "poisoning_rate": rate,
            "selection_seed": selection_seed,
            "poison_count": poison_count,
            "total_samples": len(scoped_df)
        }

        rate_dir = f"rate_{rate:.2f}"
        manifest_path = save_poisoned_data(poisoned_df, output_root / rate_dir, metadata)

        print(f"Rate: {rate:.0%} | Số mẫu đổi nhãn: {poison_count}/{len(scoped_df)} | Đã lưu: {manifest_path.name}")

if __name__ == "__main__":
    
    CONFIG_PATH = "configs/attacks/Label_flipping/Label_flipping.yaml"
    main("E-LF1", CONFIG_PATH)