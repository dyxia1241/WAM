from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def runtime_info() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def write_manifest(
    path: str | Path,
    *,
    kind: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    experiment: str,
    checkpoint: str | None = None,
    split: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "experiment": experiment,
        "split": split,
        "checkpoint": checkpoint,
        "git_commit": git_commit(repo_root),
        "runtime": runtime_info(),
        "config": config,
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload

