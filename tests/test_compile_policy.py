"""Formal compile policy, tested without invoking a compiler or using a GPU."""

import os
import subprocess
import sys

import pytest
import torch
import yaml

from h9dp import engine
from h9dp.common import load_config


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_formal_training_rejects_eager_before_touching_data_or_gpu(monkeypatch, value):
    def unexpected(*args):
        pytest.fail("Disabled compile must be rejected before reading data")
    monkeypatch.setattr(engine, "load_cache", unexpected)
    config = {"training": {} if value is None else {"torch_compile": value}}
    with pytest.raises(ValueError, match="requires.*torch_compile=true"):
        engine.train(config)


def test_enabled_policy_reaches_existing_data_checks(monkeypatch):
    def sentinel(config):
        raise RuntimeError("existing data checks reached")
    monkeypatch.setattr(engine, "load_cache", sentinel)
    with pytest.raises(RuntimeError, match="existing data checks reached"):
        engine.train({"training": {"torch_compile": True}})


def test_compile_wrapper_uses_raw_model_and_does_not_suppress_failures(monkeypatch):
    from torch._dynamo import config as dynamo_config

    model = torch.nn.Linear(2, 1)
    calls = []
    monkeypatch.setattr(dynamo_config, "suppress_errors", True)
    def compiler(value, **options):
        calls.append((value, options))
        return "compiled-forward"
    monkeypatch.setattr(torch, "compile", compiler)
    config = {"training": {"torch_compile": True, "compile_mode": "default"}}
    assert engine.compile_model(model, config) == "compiled-forward"
    assert calls == [(model, {"mode": "default", "dynamic": False})]
    assert dynamo_config.suppress_errors is False


def test_compiler_failure_is_not_caught_as_eager_fallback(monkeypatch):
    from torch._dynamo import config as dynamo_config

    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    def compiler(*args, **kwargs):
        raise RuntimeError("compile failed")
    monkeypatch.setattr(torch, "compile", compiler)
    with pytest.raises(RuntimeError, match="compile failed"):
        engine.compile_model(torch.nn.Linear(1, 1), {"training": {"torch_compile": True}})


def test_diagnostic_helper_can_still_deliberately_use_eager():
    model = torch.nn.Linear(1, 1)
    assert engine.compile_model(model, {"training": {"torch_compile": False}}) is model


def test_formal_default_configs_enable_compile():
    for path in ("configs/train.yaml", "configs/t1a.yaml"):
        assert load_config(path)["training"]["torch_compile"] is True


def test_formal_cli_rejects_eager_without_creating_run_or_accessing_gpu(tmp_path):
    config = load_config("configs/t1a.yaml")
    config["training"]["torch_compile"] = False
    path, run = tmp_path / "eager.yaml", tmp_path / "must_not_start"
    path.write_text(yaml.safe_dump(config))
    environment = dict(os.environ, MKL_THREADING_LAYER="GNU", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(
        [sys.executable, "scripts/h9dp.py", "train", "--config", str(path), "--run-dir", str(run)],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert result.returncode != 0
    assert "requires training.torch_compile=true" in result.stderr
    assert not run.exists()
