import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict

from src.data.labels import LABELS

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "checkpoints"
    / "baseline"
    / "model.onnx"
)
session = ort.InferenceSession(str(MODEL_PATH))
CONFIDENCE_THRESHOLD = 0.6


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def predict(input_image: np.ndarray, top_k: int = 3) -> Dict:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: input_image})
    logits = outputs[0]                 
    probs = softmax(logits)[0]         

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_values = probs[top_indices]

    top_k_results = [
        {"label": LABELS[idx], "confidence": float(prob)}
        for idx, prob in zip(top_indices, top_values)
    ]

    status = "ok" if top_k_results[0]["confidence"] >= CONFIDENCE_THRESHOLD else "low_confidence"

    return {
        "predicted_label": top_k_results[0]["label"],
        "confidence": top_k_results[0]["confidence"],
        "top_k": top_k_results,
        "status": status
    }