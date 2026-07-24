"""Test that a run manifest captures the fields reproducibility depends on and records failures."""

import json

import pytest
from src.common.config import Config
from src.common.manifest import RunManifest


def test_manifest_written_with_core_fields(config_with_runs_in_tmp: Config) -> None:
    cfg = config_with_runs_in_tmp
    with RunManifest(cfg, script="tests/dummy.py") as run:
        run.add_model("qwen2.5:7b-instruct-q4_K_M")
        run.note("candidates_generated", 3)
        run_dir = run.run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "config_hash", "seed", "packages", "wall_clock_seconds", "models"):
        assert key in manifest
    assert manifest["seed"] == 42
    assert manifest["models"] == ["qwen2.5:7b-instruct-q4_K_M"]
    assert manifest["candidates_generated"] == 3
    assert manifest["error"] is None
    assert manifest["packages"]  # non-empty: pip-freeze equivalent captured


def test_manifest_records_error_on_crash(config_with_runs_in_tmp: Config) -> None:
    cfg = config_with_runs_in_tmp
    with pytest.raises(ValueError), RunManifest(cfg, script="tests/dummy.py") as run:
        run_dir = run.run_dir
        raise ValueError("boom")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # A crash must still leave a manifest, with the error recorded rather than swallowed.
    assert "ValueError" in manifest["error"]
