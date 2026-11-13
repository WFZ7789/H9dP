"""Memory-mapped native500 sequences and synchronized augmentation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def augment_sequence(images: np.ndarray, mask: np.ndarray, centers: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Geometry is shared by all frames/bands; coordinates are augmentation-only."""
    height, width = images.shape[-2:]
    if np.random.random() < config["rotation_probability"]:
        images = images[..., ::-1, ::-1].copy()
        mask = mask[..., ::-1, ::-1].copy()
        centers = np.asarray([width - 1, height - 1]) - centers
    if np.random.random() < config["translation_probability"]:
        bound = config["maximum_translation"]
        lower_x = max(-bound, math.ceil(-centers[:, 0].min()))
        upper_x = min(bound, math.floor(width - 1 - centers[:, 0].max()))
        lower_y = max(-bound, math.ceil(-centers[:, 1].min()))
        upper_y = min(bound, math.floor(height - 1 - centers[:, 1].max()))
        if lower_x <= upper_x and lower_y <= upper_y:
            dx = int(np.random.randint(lower_x, upper_x + 1))
            dy = int(np.random.randint(lower_y, upper_y + 1))
            moved_images = np.zeros_like(images)
            moved_mask = np.zeros_like(mask)
            src_x = slice(max(0, -dx), min(width, width - dx))
            src_y = slice(max(0, -dy), min(height, height - dy))
            dst_x = slice(max(0, dx), min(width, width + dx))
            dst_y = slice(max(0, dy), min(height, height + dy))
            moved_images[..., dst_y, dst_x] = images[..., src_y, src_x]
            moved_mask[..., dst_y, dst_x] = mask[..., src_y, src_x]
            images, mask = moved_images, moved_mask
    return images, mask


class SequenceDataset(Dataset):
    def __init__(self, directory: str | Path, index: dict, split: str, augmentation: dict | None = None):
        self.directory = Path(directory)
        self.samples = [sample for sample in index["samples"] if sample["split"] == split]
        if len({sample["record"] for sample in index["samples"]}) != len(index["samples"]):
            raise ValueError("Every record must be a distinct sample, including across splits")
        self.mean = np.asarray(index["normalization"]["mean_kelvin"], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(index["normalization"]["std_kelvin"], dtype=np.float32)[None, :, None, None]
        self.augmentation = augmentation
        self._images = None
        self._packed = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_packed"] = None
        return state

    def __getitem__(self, position: int) -> dict:
        if self._images is None or self._packed is None:
            self._images = np.load(self.directory / "brightness.npy", mmap_mode="r")
            self._packed = np.load(self.directory / "validity.npy", mmap_mode="r")
        images_source = self._images
        packed_source = self._packed
        if images_source is None or packed_source is None:
            raise RuntimeError("Dataset arrays failed to load")
        sample = self.samples[position]
        images = np.array(images_source[sample["frames"]], dtype=np.float32, copy=True)
        masks = np.unpackbits(packed_source[sample["frames"]], axis=-1, count=500 * 500, bitorder="little").reshape(6, 4, 500, 500).astype(bool)
        common = np.all(masks, axis=1, keepdims=True)
        if not np.all(common.any(axis=(1, 2, 3))):
            raise ValueError(f"Entirely invalid frame in record {sample['record']}")
        if self.augmentation is not None:
            images, common = augment_sequence(images, common, np.asarray(sample["centers"]), self.augmentation)
            noise = self.augmentation["noise_kelvin"]
            if noise:
                images += np.random.normal(0, noise, images.shape).astype(np.float32)
        images -= self.mean
        images /= self.std
        images *= common
        return {
            "images": torch.from_numpy(np.ascontiguousarray(images)),
            "mask": torch.from_numpy(np.ascontiguousarray(common)),
            "time_offsets": torch.tensor(sample["time_offsets_minutes"], dtype=torch.float32),
            "target": torch.tensor(sample["dP"], dtype=torch.float32),
            "record": sample["record"],
            "quality": sample["quality"],
            "storm_id": sample["storm_id"],
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(config: dict, index: dict, split: str, batch_size: int, epoch: int = 0, augment: bool | None = None) -> DataLoader:
    training = split == "train" and augment is not False
    dataset = SequenceDataset(config["data"]["cache_dir"], index, split, config["augmentation"] if training else None)
    generator = torch.Generator().manual_seed(config["training"]["seed"] + epoch)
    workers = config["training"]["workers"]
    kwargs = {}
    if workers:
        kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": config["training"]["prefetch_factor"]}
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training, drop_last=False,
        num_workers=workers, pin_memory=config["training"]["pin_memory"],
        worker_init_fn=seed_worker, generator=generator, **kwargs,
    )

