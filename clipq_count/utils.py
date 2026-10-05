import logging
import os
import random

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("clipq_count")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def move_batch(batch, device):
    """Move one collated batch to `device`. Returns a dict for readability."""
    images, examples, text, points, counts, log_counts, density, names = batch
    return {
        "images": images.to(device, non_blocking=True),
        "examples": examples.to(device, non_blocking=True),
        "text": {k: v.to(device, non_blocking=True) for k, v in text.items()},
        "points": [p.to(device, non_blocking=True) for p in points],
        "counts": counts.to(device, non_blocking=True).float(),
        "log_counts": log_counts.to(device, non_blocking=True).float(),
        "density": density.to(device, non_blocking=True),
        "names": names,
    }
