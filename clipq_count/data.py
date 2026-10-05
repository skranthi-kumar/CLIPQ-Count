"""FSC-147 dataset with CLIP preprocessing, exemplar crops, text prompts and density maps."""
import json
import math
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import CLIPProcessor

from .utils import logger

CLIP_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224
FSC147_IMAGE_SIZE = 384.0  # annotation coordinates are in 384-px image space
MAX_COUNT = 1000


def verify_dataset(data_dir: str, annotation_file: str, split_file: str) -> bool:
    """Check that the FSC-147 files exist and every split is non-empty."""
    if not all(os.path.exists(p) for p in (data_dir, annotation_file, split_file)):
        logger.error("Dataset verification failed: missing files")
        return False
    with open(split_file) as f:
        splits = json.load(f)
    for split in ("train", "val", "test"):
        if not splits.get(split):
            logger.error("Missing or empty %s split", split)
            return False
    logger.info("Dataset verification passed")
    return True


def generate_density_map(points: torch.Tensor, size=(IMAGE_SIZE, IMAGE_SIZE), sigma: float = 15.0) -> torch.Tensor:
    """Render normalised point annotations (in [0,1]) as a Gaussian density map that sums to 1."""
    density = torch.zeros(size)
    if len(points) == 0:
        return density
    points = points * torch.tensor([size[0], size[1]])
    ys = torch.arange(size[0]).view(-1, 1)
    xs = torch.arange(size[1]).view(1, -1)
    for point in points:
        x, y = point.long()
        if 0 <= x < size[0] and 0 <= y < size[1]:
            density += torch.exp(-((ys - int(x)) ** 2 + (xs - int(y)) ** 2) / (2 * sigma ** 2))
    return density / (density.sum() + 1e-6)


class FSC147Dataset(Dataset):
    """One sample = (image, k exemplar crops, text prompt, points, count, log-count, density map, name)."""

    def __init__(self, data_dir, annotation_file, split_file, shots=3, split="train", augment=True):
        for path in (data_dir, annotation_file, split_file):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        self.data_dir = data_dir
        self.split = split
        self.shots = min(shots, 5)
        self.augment = augment and split == "train"

        with open(split_file) as f:
            self.image_files = json.load(f).get(split, [])
        if not self.image_files:
            raise ValueError(f"No images found for split: {split}")
        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.transform = (
            T.Compose([
                T.RandomRotation(15),
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ])
            if self.augment
            else T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE))])
        )
        self.processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        logger.info("Loaded %d images for %s split", len(self.image_files), split)

    def __len__(self):
        return len(self.image_files)

    # ---- helpers -------------------------------------------------------
    def _load_image(self, path, fallback=None):
        try:
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        except Exception as exc:  # corrupt file
            logger.warning("Failed to load %s: %s", path, exc)
        return fallback or Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="gray")

    def _find_exemplar(self, example_path, main_image):
        base = os.path.basename(example_path)
        candidates = [
            example_path,
            os.path.join(self.data_dir, base),
            os.path.join(self.data_dir, "box_examples", base),
            os.path.join(self.data_dir, base.split("_")[0] + ".jpg"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return self._load_image(path, main_image)
        return main_image

    def _clip_pixels(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # ---- main ----------------------------------------------------------
    def __getitem__(self, idx):
        name = self.image_files[idx]
        anno = self.annotations.get(name, {"points": [], "box_examples_path": [], "class": "unknown"})

        image = self.transform(self._load_image(os.path.join(self.data_dir, name)))
        image_pixels = self._clip_pixels(image)

        count = min(len(anno["points"]), MAX_COUNT)
        log_count = math.log(count + 1)

        class_name = anno.get("class", "objects").lower()
        prompt = f"A photo containing multiple {class_name} in a cluttered scene"
        text = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)

        pts = anno.get("points", [])
        points = torch.tensor(pts, dtype=torch.float32) / FSC147_IMAGE_SIZE if pts else torch.zeros((0, 2))
        density = generate_density_map(points)

        exemplar_paths = anno.get("box_examples_path", [])
        if self.augment and len(exemplar_paths) > self.shots:
            exemplar_paths = random.sample(exemplar_paths, self.shots)
        exemplars = [self._clip_pixels(self.transform(self._find_exemplar(p, image))) for p in exemplar_paths[: self.shots]]
        while len(exemplars) < self.shots:  # pad with the query image itself
            exemplars.append(image_pixels)
        exemplars = torch.stack(exemplars)

        return image_pixels, exemplars, text, points, count, log_count, density, name


def collate_fn(batch):
    images, exemplars, texts, points, counts, log_counts, densities, names = zip(*batch)
    text = {
        "input_ids": default_collate([t["input_ids"] for t in texts]).squeeze(1),
        "attention_mask": default_collate([t["attention_mask"] for t in texts]).squeeze(1),
    }
    return (
        default_collate(images),
        default_collate(exemplars),
        text,
        list(points),
        default_collate(counts),
        default_collate(log_counts),
        default_collate(densities),
        list(names),
    )
