"""Run manifests: a tamper-evident record of exactly how each execution was produced.

Charter RULE 6 requires every script to drop a manifest under ``runs/<timestamp>/manifest.json``
capturing git SHA, UTC timestamps, config hash, installed package versions, model tags, seed,
input file hashes, and wall-clock duration. If a result can't be tied back to one of these, it
isn't reproducible and, for this lab, doesn't count.

Usage::

    with RunManifest(cfg, script="src/data/prices.py") as run:
        run.add_input(Path("data/raw/bhavcopy_2020.parquet"))
        run.add_model("qwen2.5:7b-instruct-q4_K_M")
        ...  # do the work
    # manifest.json is written on exit, including duration and any error that escaped
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any

from src.common.config import Config
from src.common.io import sha256_file
from src.common.log import get_logger

_log = get_logger(__name__)


def _git_sha() -> str:
    """Current commit SHA, or a sentinel if we're not in a git tree / git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        sha = out.stdout.strip()
        # A dirty tree means the recorded SHA doesn't fully describe the code that ran; flag it.
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - manifest must record *something*, never crash the run
        return "unknown"


def _installed_packages() -> dict[str, str]:
    """Map of installed distribution name -> version, the equivalent of ``pip freeze``."""
    return {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}


class RunManifest:
    """Context manager that assembles and writes one run's manifest."""

    def __init__(self, cfg: Config, script: str) -> None:
        self._cfg = cfg
        self._script = script
        self._start_utc = datetime.now(UTC)
        # A microsecond-stamped directory keeps concurrent runs from colliding.
        stamp = self._start_utc.strftime("%Y%m%dT%H%M%S_%f")
        self.run_dir = cfg.paths.runs / stamp
        self._inputs: dict[str, str] = {}
        self._models: list[str] = []
        self._extra: dict[str, Any] = {}

    def add_input(self, path: Path) -> None:
        """Record an input file by hash so the run is pinned to the exact bytes it consumed."""
        self._inputs[str(path)] = sha256_file(path)

    def add_model(self, tag: str) -> None:
        """Record a local model tag (e.g. an Ollama tag). Results aren't reproducible without it."""
        self._models.append(tag)

    def note(self, key: str, value: Any) -> None:
        """Attach an arbitrary extra field to the manifest."""
        self._extra[key] = value

    def __enter__(self) -> RunManifest:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _log.info("run started: %s -> %s", self._script, self.run_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        end_utc = datetime.now(UTC)
        manifest = {
            "script": self._script,
            "git_sha": _git_sha(),
            "started_utc": self._start_utc.isoformat(),
            "ended_utc": end_utc.isoformat(),
            "wall_clock_seconds": (end_utc - self._start_utc).total_seconds(),
            "config_hash": self._cfg.config_hash,
            "config_path": str(self._cfg.source_path),
            "seed": self._cfg.meta.seed,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "models": self._models,
            "input_hashes": self._inputs,
            "packages": _installed_packages(),
            "error": None if exc is None else f"{exc_type.__name__}: {exc}",
            **self._extra,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log.info("run finished in %.1fs; manifest at %s/manifest.json",
                  manifest["wall_clock_seconds"], self.run_dir)
