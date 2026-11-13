import numpy as np
import pytest
import torch

from h9dp.common import load_config
from h9dp.dataset import SequenceDataset, augment_sequence
from h9dp.model import DensePressureRegressor, pack_observations, regression_loss
from h9dp.prepare import select_validation_records, split_for


def test_single_scalar_and_invalid_pixels():
    torch.set_num_threads(2)
    config = load_config("configs/train.yaml")
    config["model"]["channels"] = 8
    config["model"]["head_width"] = 16
    model = DensePressureRegressor(config["model"]).eval()
    shapes = []
    handles = [block.register_forward_hook(lambda module, args, output: shapes.append(output.shape[-2:])) for block in model.blocks]
    images = torch.randn(1, 6, 4, 500, 500)
    mask = torch.ones(1, 6, 1, 500, 500, dtype=torch.bool)
    mask[..., :8, :] = False
    times = torch.tensor([[-300., -180., -60., -30., -10., 0.]])
    altered = images.clone()
    altered[..., :8, :] = 1e6
    with torch.no_grad():
        reference = model(images, times, mask)
        prediction = model(altered, times, mask)
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all() and (prediction > 0).all()
    assert len(model.blocks) == 16 and set(shapes) == {(250, 250)}
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(reference, prediction)
    with torch.no_grad(), pytest.raises(ValueError, match="invalid"):
        model(images, times, torch.zeros_like(mask))


def test_record_equal_loss_and_fixed_record_split():
    result = regression_loss(torch.tensor([60., 80., 60., 80.]), torch.full((4,), 70.), 20.)
    torch.testing.assert_close(result, torch.full((4,), 0.25))
    config = load_config("configs/train.yaml")
    # These can all be records of one cyclone; identity is exclusively record ID.
    records = [(0, 2020), (1, 2025), *[(i, 2026) for i in range(2, 10)]]
    selected = select_validation_records(records, config)
    assert len(selected) == 4
    assert selected == select_validation_records(list(reversed(records)), config)
    assert split_for(0, 2020, config, selected) == "train"
    assert {split_for(i, 2026, config, selected) for i in range(2, 10)} == {"train", "validation"}
    with pytest.raises(ValueError, match="Duplicate record"):
        select_validation_records(records + [(2, 2026)], config)


def test_pixel_rearrangement_preserves_every_valid_observation():
    images = torch.arange(6*4*8*8, dtype=torch.float32).reshape(1, 6, 4, 8, 8)
    mask = torch.ones((1, 6, 1, 8, 8), dtype=torch.bool)
    mask[..., :2, :2] = False
    times = torch.tensor([[-300., -180., -60., -30., -10., 0.]])
    packed, valid = pack_observations(images, times, mask)
    restored = torch.nn.functional.pixel_shuffle(packed[:, :96], 2)
    torch.testing.assert_close(restored, images.masked_fill(~mask, 0).reshape(1, 24, 8, 8))
    restored_masks = torch.nn.functional.pixel_shuffle(packed[:, 96:120], 2)
    torch.testing.assert_close(restored_masks, mask[:, :, 0].float())
    torch.testing.assert_close(packed[:, 120:, 0, 0], times/300)
    assert not valid[0, 0, 0, 0]


def test_memmap_mask_normalization_and_augmentation(tmp_path):
    images = np.full((1, 4, 500, 500), 250, dtype=np.float32)
    valid = np.ones_like(images, dtype=bool)
    valid[..., 0, 0] = False
    np.save(tmp_path / "brightness.npy", images)
    np.save(tmp_path / "validity.npy", np.packbits(valid.reshape(1, 4, -1), axis=-1, bitorder="little"))
    sample = {"split": "train", "frames": [0]*6, "centers": [[250., 250.]]*6, "time_offsets_minutes": [-300,-180,-60,-30,-10,0], "dP": 70., "record": 0, "quality": "Good", "storm_id": "2020:Example"}
    index = {"samples": [sample], "normalization": {"mean_kelvin": [240]*4, "std_kelvin": [20]*4}}
    item = SequenceDataset(tmp_path, index, "train")[0]
    assert item["images"].shape == (6, 4, 500, 500)
    assert torch.all(item["images"][..., 0, 0] == 0)
    assert item["images"][0, 0, 1, 1] == 0.5
    assert "centers" not in item
    np.random.seed(42)
    moved, moved_mask = augment_sequence(item["images"].numpy(), item["mask"].numpy(), np.asarray(sample["centers"]), {"rotation_probability": 1, "translation_probability": 1, "maximum_translation": 64})
    assert moved.shape == (6, 4, 500, 500)
    assert np.array_equal(moved[0], moved[5])
    assert np.array_equal(moved_mask[0], moved_mask[5])
