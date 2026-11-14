import os
import pandas as pd
import json

# Tạo thư mục data/splits nếu chưa có
os.makedirs("data/splits", exist_ok=True)

# 1. Tạo file class_to_idx.json (Giả lập 3 nhãn: mèo, chó, chim)
class_mapping = {"cat": 0, "dog": 1, "bird": 2}
with open("data/splits/class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(class_mapping, f, indent=4)

# 2. Tạo file manifest_train.csv (Giả lập 100 mẫu để test các tỷ lệ 1%, 5%, 10%...)
train_data = []
for i in range(100):
    train_data.append({
        "sample_id": f"train_{i:03d}",
        "file_path": f"images/train/img_{i:03d}.jpg",
        "label": i % 3,         # Xoay vòng nhãn 0, 1, 2
        "in_subset": True       # Cột này cần cho scope "subset"
    })
pd.DataFrame(train_data).to_csv("data/splits/manifest_train.csv", index=False)

# 3. Tạo file manifest_test.csv (Giả lập 20 mẫu)
test_data = []
for i in range(20):
    test_data.append({
        "sample_id": f"test_{i:03d}",
        "file_path": f"images/test/img_{i:03d}.jpg",
        "label": i % 3
    })
pd.DataFrame(test_data).to_csv("data/splits/manifest_test.csv", index=False)

print("Đã tạo xong dữ liệu giả tại thư mục data/splits/")