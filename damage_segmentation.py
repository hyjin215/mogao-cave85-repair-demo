"""U-Net++ damage segmentation model and tiled inference for Dunhuang murals."""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Missing dependency: segmentation-models-pytorch. Run: "
        "pip install segmentation-models-pytorch"
    ) from exc

IMAGE_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_unetplusplus(pretrained_encoder: bool = False):
    """Create a binary U-Net++ model with a ResNet-34 encoder."""
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_torch_load(path: Path, device: torch.device):
    # weights_only exists in newer PyTorch versions; support older versions too.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location=device)


def load_damage_model(checkpoint_path: Path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"U-Net++ checkpoint not found: {checkpoint_path}")

    device = device or choose_device()
    checkpoint = _safe_torch_load(checkpoint_path, device)
    model = build_unetplusplus(pretrained_encoder=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def _tile_starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_damage_mask(
    image_bgr: np.ndarray,
    checkpoint_path: Path,
    threshold: float = 0.50,
    tile_size: int = IMAGE_SIZE,
    overlap: int = 96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (binary_mask_0_255, probability_map_0_255) at original image size.

    Large images are processed in overlapping 512x512 tiles to avoid shrinking
    fine cracks and flaking into a single 512x512 image.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a valid 3-channel BGR image")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")

    model, device = load_damage_model(checkpoint_path)
    original_h, original_w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pad_h = max(0, tile_size - original_h)
    pad_w = max(0, tile_size - original_w)
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(
            rgb, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
        )

    padded_h, padded_w = rgb.shape[:2]
    stride = tile_size - overlap
    ys = _tile_starts(padded_h, tile_size, stride)
    xs = _tile_starts(padded_w, tile_size, stride)

    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    prob_count = np.zeros((padded_h, padded_w), dtype=np.float32)

    with torch.inference_mode():
        for y in ys:
            for x in xs:
                tile = rgb[y:y + tile_size, x:x + tile_size].astype(np.float32) / 255.0
                tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
                tensor = torch.from_numpy(tile.transpose(2, 0, 1).copy())
                tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
                logits = model(tensor)
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prob_sum[y:y + tile_size, x:x + tile_size] += probability
                prob_count[y:y + tile_size, x:x + tile_size] += 1.0

    probability_map = prob_sum / np.maximum(prob_count, 1.0)
    probability_map = probability_map[:original_h, :original_w]
    binary_mask = (probability_map >= threshold).astype(np.uint8) * 255
    probability_8bit = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
    return binary_mask, probability_8bit
