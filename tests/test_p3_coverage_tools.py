"""Geometry and routing reports use synthetic metadata, not archive contents."""

import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_training_geometry_excludes_validation_and_counts_original_frames():
    geometry_report = runpy.run_path(str(ROOT / "scripts/check_p3_coverage.py"))["geometry_report"]
    index = {"samples": [
        {"split": "train", "centers": [[250., 250.]] * 6, "sources": ["R3"] * 6},
        {"split": "train", "centers": [[450., 450.]] * 6, "sources": ["FLDK"] * 6},
        {"split": "validation", "centers": [[0., 0.]] * 6, "sources": ["R3"] * 6},
    ]}
    report = geometry_report(index, 129, 32)
    assert report["sources"]["R3"]["frame_occurrences"] == 6
    assert report["sources"]["R3"]["unaugmented"] == 1.
    assert report["sources"]["R3"]["augmented_expected"] == 1.
    assert report["sources"]["FLDK"]["augmented_expected"] == 0.
    assert report["window_bounds_inclusive"] == [186, 314]


def test_routing_report_distinguishes_fixed_and_offcenter_dynamic_windows():
    route_statistics = runpy.run_path(str(ROOT / "scripts/diagnose_p3_coverage.py"))["route_statistics"]
    references = np.full((2, 6, 2), 450.)
    centers = np.full((2, 2, 4, 6, 2), 450 / 249.5 - 1)
    centers[:, 1] -= 10 / 249.5
    radii = np.full((2, 2, 4, 6, 1), 16.)
    report = route_statistics(centers, radii, references, 25, 129)
    assert report["fixed_window_reference_fraction"] == 0.
    assert report["layers"][0]["reference_in_any_native_window_all_frames_fraction"] == 1.
    assert report["layers"][0]["nearest_reader_distance_latest_pixels_median"] < 1e-8
    assert np.isclose(report["layers"][1]["max_axis_refinement_pixels"], 10.)


def test_cpu_validation_aggregates_partial_batches_and_read_summaries():
    import torch
    from torch.utils.data import DataLoader, Dataset

    from h9dp.engine import validate

    class SyntheticRecords(Dataset):
        def __init__(self):
            self.samples = [{"record": i, "sources": ["R3"] * 6} for i in range(3)]

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"record": index, "target": 40. + index * 20., "quality": "Good", "storm_id": "synthetic"}

    class SyntheticModel(torch.nn.Module):
        def forward(self, images, time_offsets, mask):
            del images, time_offsets, mask
            raise AssertionError("Use the patched forward in this CPU engine test")

    # Engine's production forward deliberately uses CUDA BF16. Patch only the
    # execution boundary, keeping coverage/count/summary aggregation unchanged.
    from unittest.mock import patch

    model = SyntheticModel()

    def fake_forward(model, batch):
        model.last_read_summary = {"source_mass": [[.1, .7, .1, .1]], "radius": 16.}
        return batch["target"] + 2.

    with patch("h9dp.engine._forward", fake_forward):
        metrics, rows = validate(model, DataLoader(SyntheticRecords(), batch_size=2), torch.device("cpu"))
    assert metrics["count"] == len(rows) == 3
    assert metrics["rmse"] == 2.
    assert metrics["groups"]["R3-only"]["count"] == 3
    assert metrics["groups"]["FLDK-containing"]["rmse"] is None
    np.testing.assert_allclose(metrics["reads"]["source_mass"], [[.1, .7, .1, .1]])
    assert metrics["reads"]["radius"] == 16.
