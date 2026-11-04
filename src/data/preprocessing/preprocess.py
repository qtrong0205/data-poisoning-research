import io
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Converts raw uploaded image bytes into a preprocessed numpy array
    ready for ONNX inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor.numpy().astype(np.float32)