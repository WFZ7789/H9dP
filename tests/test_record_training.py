import copy

import numpy as np
import pytest

from h9dp.common import load_config
from h9dp.dataset import SequenceDataset, make_loader
from h9dp.prepare import update_training_statistics


def test_uniform_record_sampling_covers_every_record_once(tmp_path):
    config = load_config("configs/train.yaml")
    config["data"]["cache_dir"] = str(tmp_path)
    config["training"]["workers"] = 0
    samples = [{"record": i, "split": "train", "storm_id": "2026:SameCyclone"} for i in range(12)]
    index = {"samples": samples, "normalization": {"mean_kelvin": [250]*4, "std_kelvin": [20]*4}}
    loader = make_loader(config, index, "train", 4, epoch=3)
    draws = list(loader.sampler)
    assert len(draws) == len(set(draws)) == len(samples)
    assert sorted(draws) == list(range(len(samples)))
    assert draws == list(make_loader(config, index, "train", 4, epoch=3).sampler)
    plain = make_loader(config, index, "train", 4, augment=False)
    assert list(plain.sampler) == list(range(len(samples)))
    duplicate = copy.deepcopy(index)
    duplicate["samples"].append({"record": 0, "split": "validation", "storm_id": "2026:SameCyclone"})
    with pytest.raises(ValueError, match="distinct sample"):
        SequenceDataset(tmp_path, duplicate, "train")


def test_normalization_uses_equal_records_and_excludes_validation(tmp_path):
    images = np.stack([np.full((4, 500, 500), value, dtype=np.float32) for value in (250, 262, 350)])
    validity = np.ones((3, 4, 500*500), dtype=np.uint8)
    np.save(tmp_path / "brightness.npy", images)
    np.save(tmp_path / "validity.npy", np.packbits(validity, axis=-1, bitorder="little"))
    index = {"samples": [
        {"record": 1, "split": "train", "frames": [0]*6, "dP": 70.},
        {"record": 2, "split": "train", "frames": [0]*5+[1], "dP": 90.},
        {"record": 3, "split": "validation", "frames": [2]*6, "dP": 200.},
    ], "summary": {}}
    update_training_statistics(tmp_path, index)
    np.testing.assert_allclose(index["normalization"]["mean_kelvin"], [251]*4)
    np.testing.assert_allclose(index["normalization"]["std_kelvin"], [np.sqrt(11)]*4)
    assert index["normalization"]["training_frame_occurrences"] == 12
    assert index["target_normalization"]["mean_hpa"] == 80
    assert index["target_normalization"]["std_hpa"] == 10
