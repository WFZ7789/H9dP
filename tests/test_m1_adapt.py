"""Small synthetic checks; no real-data training or archive extraction."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from h9dp.common import load_config
from h9dp.engine import seed_everything
from h9dp.m1 import M5, M1Encoder
from h9dp.m1_adapt import (
    CalibratedM5,
    JointM1,
    design_matrix,
    grouped_folds,
    joint_loss,
    learning_rate,
    normalization,
    quality_weights,
)
from scripts import m1_adapt
from scripts.diagnose_m1_adapt import diagnose
from scripts.m1 import record_arrays
from scripts.m1_adapt import padded_batch


def test_rotating_folds_cover_every_record_without_splitting_storms():
    records = [{"storm_id": f"2020:s{i // 3}"} for i in range(30)]
    folds = grouped_folds(records, 3, 1)
    np.testing.assert_array_equal(folds, grouped_folds(records, 3, 1))
    assert set(folds) == {0, 1, 2}
    trained, evaluated = set(), set()
    for fold in range(3):
        train = np.flatnonzero(folds != fold)
        evaluation = np.flatnonzero(folds == fold)
        assert {records[i]["storm_id"] for i in train}.isdisjoint(
            records[i]["storm_id"] for i in evaluation
        )
        trained.update(train)
        evaluated.update(evaluation)
    assert trained == evaluated == set(range(30))
    # Inner calibration folds must not introduce any outer evaluation storm.
    outer_train = np.flatnonzero(folds != 0)
    inner = grouped_folds([records[i] for i in outer_train], 2, 1)
    assert len(inner) == len(outer_train)
    assert all(folds[i] != 0 for i in outer_train)


def test_normalization_ignores_missing_frames_and_other_folds():
    records = [
        {"stats": [[[10.0, 100.0, 1.0]], [[0.0, 0.0, 0.0]]]},
        {"stats": [[[1000.0, 1000000.0, 1.0]]]},
    ]
    mean, std = normalization(records, [0])
    np.testing.assert_allclose(mean, [10.0])
    assert std[0] > 0


def test_missing_frame_statistics_are_zero_not_fake_kelvin():
    image = np.full((2, 1, 500, 500), 250.0, np.float32)
    mask = np.ones((2, 1, 500, 500), bool)
    mask[1] = False
    sample = {"centers": [[250, 250], [250, 250]]}
    with pytest.raises(ValueError, match="no valid pixels"):
        record_arrays(sample, image, mask, (32, 96))
    _, numeric, coverage, stats = record_arrays(sample, image, mask, (32, 96), allow_missing=True)
    assert not numeric[1].any()
    assert not coverage[1].any()
    assert not stats[1].any()
    assert stats[0, 0, 2] == 250000


def test_encoder_does_not_normalize_empty_region_placeholders():
    seed_everything(1)
    encoder = M1Encoder(bands=4, frames=2, image_size=32, radii=(4, 10),
                        channels=(8, 12, 24, 32, 32), frame_width=8).eval()
    encoder.band_mean.fill_(250.0)
    encoder.band_std.fill_(10.0)
    images = torch.full((1, 2, 4, 32, 32), 250.0)
    mask = torch.ones((1, 2, 1, 32, 32), dtype=torch.bool)
    mask[:, 1] = False
    numeric = torch.full((1, 2, 4, 3, 3), 250.0)
    area = torch.ones((1, 2, 3), dtype=torch.int32)
    area[:, 1] = 0
    centers = torch.full((1, 2, 2), 16.0)
    with torch.no_grad():
        numeric[:, 1] = 0
        first = encoder(images, mask, numeric, area, centers)
        numeric[:, 1] = 10000
        second = encoder(images, mask, numeric, area, centers)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(second).all()


def test_calibrator_learns_from_predicted_not_true_parameter_distribution():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(100, 5))
    truth[:, 4] = -70 + truth[:, 4]
    coefficients = np.array([50.0, 3.0, 6.0, 70.0, 3.0])
    y = design_matrix(truth) @ coefficients
    teacher = M5(coefficients)
    shrunk = truth.copy()
    shrunk[:, :4] *= 0.4
    calibrator = CalibratedM5.fit(shrunk, y, np.ones(100), teacher, alpha=1.0)
    assert np.mean((calibrator.predict(shrunk) - y) ** 2) < np.mean((teacher.predict(shrunk) - y) ** 2)
    with pytest.raises(ValueError):
        CalibratedM5.fit(shrunk, y, np.zeros(100), teacher, 1.0)


def test_weights_never_exclude_records_and_schedule_is_replayable():
    records = [{"quality": q} for q in ["Good", "Fair", "Poor"]]
    np.testing.assert_allclose(quality_weights(records, {"Good": 2, "Fair": 1, "Poor": 0.5}), [2, 1, 0.5])
    with pytest.raises(ValueError):
        quality_weights(records, {"Good": 2, "Fair": 1, "Poor": 0})
    assert learning_rate(12, 0.0003, [13, 25]) == 0.0003
    assert learning_rate(13, 0.0003, [13, 25]) == 0.00015
    assert learning_rate(25, 0.0003, [13, 25]) == 0.000075


def test_padding_preserves_actual_count_and_fixed_compiled_shape():
    batch = {"images": torch.ones((2, 6, 4, 2, 2)), "record": torch.tensor([3, 4])}
    padded, count = padded_batch(batch, torch.device("cpu"), 4)
    assert count == 2
    assert len(padded["images"]) == 4
    assert padded["record"].tolist() == [3, 4, 3, 3]


def test_adaptation_config_loads_with_repository_contract():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/m1_adapt.yaml")
    assert config["training"]["torch_compile"] is True
    assert Path(config["output_dir"]).is_absolute()
    assert min(config["training"]["quality_weights"].values()) > 0


def test_nested_probe_has_no_permanent_holdout_or_outer_storm_leakage(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    parameters = rng.normal(size=(48, 5))
    parameters[:, 4] = -70 + parameters[:, 4]
    target = design_matrix(parameters) @ np.array([50, 3, 6, 70, 3])
    records = [{"record": i, "name": f"s{i // 4}", "storm_id": f"2020:s{i // 4}", "year": 2020,
                "quality": ["Good", "Fair", "Poor"][i % 3], "origin": "synthetic", "missing_frames": 0,
                "dP": float(target[i]), "params": parameters[i].tolist()}
               for i in range(48)]
    work = tmp_path / "data"
    work.mkdir()
    (work / "all_meta.json").write_text(json.dumps({"records": records, "counts": {"all": 48}}))
    catalog = [{**r, "time": {"year": r["year"]}} for r in records]
    monkeypatch.setattr(m1_adapt, "read_catalog", lambda _: (catalog, "synthetic"))
    config = {"repository_root": str(tmp_path), "data": {"work_dir": str(work), "records": "synthetic.json"},
              "training": {"seed": 1, "quality_weights": {"Good": 2, "Fair": 1, "Poor": 0.5}},
              "crossfit": {"outer_folds": 2, "inner_folds": 2, "ridge_alpha": 10}}
    monkeypatch.setattr(m1_adapt, "initialize_device", lambda config: (torch.device("cpu"), {}))
    calls = []

    def fake_train_pack(train, evaluation, config, meta, directory, epochs, seed, device):
        assert {records[i]["storm_id"] for i in train}.isdisjoint(records[i]["storm_id"] for i in evaluation)
        calls.append((train, evaluation))
        return parameters[evaluation]

    monkeypatch.setattr(m1_adapt, "train_pack", fake_train_pack)
    result = m1_adapt.probe(config, tmp_path / "run", 1)
    assert len(calls) == 6
    outer_training = [index for train, _ in (calls[2], calls[5]) for index in train]
    outer_evaluation = [index for _, evaluation in (calls[2], calls[5]) for index in evaluation]
    assert sorted(outer_training) == sorted(outer_evaluation) == list(range(48))
    assert result["raw_m5"]["all"]["count"] == 48
    assert (tmp_path / "run" / "predictions_epoch001.csv").exists()
    diagnosis = diagnose(tmp_path / "run", 1, work)
    assert diagnosis["groups"]["paired_Good"]["oracle_paired_only"]["count"] == 16
    assert diagnosis["groups"]["paired_Good"]["oracle_paired_only"]["rmse"] < 1e-8
    assert (tmp_path / "run" / "diagnosis_epoch001.json").exists()


def test_joint_head_matches_physical_m5_and_backpropagates():
    class ConstantExpert(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(float(value)))

        def forward(self, images, *args):
            return self.value.expand(len(images), 2)

    parameters = np.array([2.0, 4.0, 0.15, 0.0, -70.0])
    coefficients = np.array([10.0, 3.0, 5.0, 70.0, 3.0])
    model = JointM1([ConstantExpert(value) for value in parameters], coefficients)
    output = model(torch.ones(2, 1), None, None, None, None, None)
    expected = M5(coefficients).predict(np.tile(parameters, (2, 1)))
    np.testing.assert_allclose(output[:, 5].detach().numpy(), expected, rtol=1e-6)
    assert sum(p.numel() for p in model.parameters()) == 5  # Head adds no parameters.
    output[:, 5].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("all_missing", [False, True])
def test_joint_loss_uses_unpaired_dp_without_nan_gradients(all_missing):
    output = torch.ones((2, 6), requires_grad=True)
    targets = torch.full((2, 5), float("nan"))
    if not all_missing:
        targets[0] = 0
    loss = joint_loss(output, targets, torch.zeros(2), torch.ones(2), torch.ones(5), 10.0, 0.25)
    loss.backward()
    assert torch.isfinite(loss)
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert (output.grad[:, 5] != 0).all()
