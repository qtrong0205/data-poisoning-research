# Demo Backdoor Evaluation (E-BD1)

Tài liệu này hướng dẫn chạy một lượt thí nghiệm Backdoor từ đầu đến cuối: chuẩn bị dữ liệu, tạo poisoned manifest, train model, đánh giá clean/ASR và xem artifacts.

## Điều kiện trước khi chạy

- Chạy các lệnh từ thư mục gốc repository.
- Python environment đã cài dependencies trong `requirements.txt`.
- Dataset PlantVillage đã được giải nén tại `data/plantvillage dataset/color` (theo `configs/dataset/dataset.yaml`).
- Máy có đủ dung lượng lưu checkpoint và artifacts. Nếu bật pretrained weights, PyTorch có thể cần tải weights ở lần chạy đầu.

## Chạy demo

### 1. Tạo và kích hoạt môi trường Python

Từ thư mục gốc repository, tạo virtual environment:
Hãy kiểm tra lại phiên bản python của bạn để có cú pháp phù hợp:

Ngoài lề: 
{
python3.10 --version

python3.10 -m venv .venv

source .venv/bin/activate

python --version }

```bash
python3 -m venv .venv
```

Kích hoạt môi trường trên macOS/Linux:

```bash
source .venv/bin/activate
```

Trên Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt, các lệnh `python` và `pip` bên dưới sẽ dùng môi trường `.venv`.

### 2. Cài dependencies

Nếu chưa cài dependencies trong môi trường hiện tại:

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị split và manifest

```bash
python scripts/prepare_dataset.py
```

Script kiểm tra split/manifests hiện có và tạo những phần còn thiếu. Dataset gốc phải ở đúng vị trí đã cấu hình.

### 4. Tạo poisoned training manifest cho E-BD1

```bash
python -c 'from src.attacks.backdoor.prepare_data import main; main("E-BD1", "configs/attacks/backdoor/backdoor.yaml")'
```

Lệnh này tạo các manifest cho poisoning rate trong cấu hình E-BD1. Mỗi manifest cùng metadata được ghi dưới:

```text
data/poisoned/backdoor/E-BD1/rate_<rate>/
```

### 5. Train và đánh giá

```bash
python -m experiments.backdoor.analysis.experiment --experiment E-BD1
```

Runner dùng manifest huấn luyện đã tạo, sau đó đánh giá model trên test set gốc và trên các test sample non-target đã gắn trigger.

### 6. Xem kết quả

Artifacts của từng poisoning rate nằm dưới:

```text
experiments/backdoor/E-BD1_poisoning_rate/rate_<rate>/size_<px>/<position>/seed_<seed>/
```

Các file chính:

- `metrics.json`: clean metrics, ASR, Per-class ASR và metadata lượt chạy.
- `clean_predictions.csv`, `asr_predictions.csv`: dự đoán từng sample.
- `clean_confusion_matrix.png`: confusion matrix trên test set sạch.
- `final_model.pth`: checkpoint đã đánh giá.
- `config_snapshot.yaml`: cấu hình và metadata của lượt chạy.
- `asr_test_manifest.csv`: test manifest non-target dùng trong ASR.

ASR là tỷ lệ sample có dự đoán bằng target label; target class được loại khỏi ASR test set và nhãn thật của các sample còn lại được giữ nguyên.

## Các experiment khác

Chỉ E-BD1 có `poisoning_rates` được điền sẵn. E-BD2, E-BD3 và E-BD4 hiện để `poisoning_rates: null`, nên cần cấu hình rates trong `configs/attacks/backdoor/backdoor.yaml` trước khi chạy. Sau đó, lặp lại bước tạo poisoned manifest với experiment ID tương ứng rồi chạy runner, ví dụ:

```bash
python -c 'from src.attacks.backdoor.prepare_data import main; main("E-BD2", "configs/attacks/backdoor/backdoor.yaml")'
python -m experiments.backdoor.analysis.experiment --experiment E-BD2
```
