from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppwam.merge_prepared_sources import read_windows


def _load_index(windows_dir: Path) -> dict[str, Any]:
    path = windows_dir / "index.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else 0.0


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else 0.0


def _series_stats(prefix: str, values: np.ndarray, eps: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_negative_rate": 0.0,
            f"{prefix}_stagnation_rate": 0.0,
            f"{prefix}_positive_rate": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_negative_rate": float(np.mean(finite < -eps)),
        f"{prefix}_stagnation_rate": float(np.mean(np.abs(finite) <= eps)),
        f"{prefix}_positive_rate": float(np.mean(finite > eps)),
    }


def _monotonicity_violations(windows: list[dict[str, Any]], phi_t: np.ndarray, eps: float) -> tuple[float, int]:
    by_episode: dict[str, list[tuple[int, int]]] = {}
    for index, window in enumerate(windows):
        by_episode.setdefault(str(window["episode_id"]), []).append((int(window["t"]), index))
    violations = 0
    comparisons = 0
    for rows in by_episode.values():
        ordered = [index for _, index in sorted(rows)]
        if len(ordered) <= 1:
            continue
        diffs = np.diff(phi_t[ordered])
        comparisons += int(diffs.shape[0])
        violations += int(np.sum(diffs < -eps))
    rate = float(violations / comparisons) if comparisons else 0.0
    return rate, violations


def _group_rows(
    windows: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    eps: float,
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, window in enumerate(windows):
        groups.setdefault(
            (
                str(window.get("split", "")),
                str(window.get("source", "")),
                str(window.get("stage", "")),
            ),
            [],
        ).append(index)
    rows: list[dict[str, float | int | str]] = []
    for (split, source, stage), indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        delta = labels["delta_phi_raw"][idx]
        row: dict[str, float | int | str] = {
            "split": split,
            "source": source,
            "stage": stage,
            "num_windows": int(idx.shape[0]),
            "phi_t_mean": _finite_mean(labels["phi_t"][idx]),
            "phi_future_mean": _finite_mean(labels["phi_future"][idx]),
            "delta_phi_raw_mean": _finite_mean(delta),
            "delta_phi_raw_std": _finite_std(delta),
            "delta_phi_raw_negative_rate": float(np.mean(delta < -eps)) if idx.size else 0.0,
            "delta_phi_raw_stagnation_rate": float(np.mean(np.abs(delta) <= eps)) if idx.size else 0.0,
            "delta_phi_raw_positive_rate": float(np.mean(delta > eps)) if idx.size else 0.0,
            "legacy_delta_phi_mean": _finite_mean(labels["delta_phi"][idx]),
        }
        rows.append(row)
    return rows


def audit_potential_gain(
    windows_dir: str | Path,
    output_dir: str | Path | None = None,
    eps: float = 1.0e-4,
) -> dict[str, float]:
    windows_path = Path(windows_dir)
    windows = read_windows(windows_path / "windows.jsonl")
    with np.load(windows_path / "labels.npz") as loaded:
        raw = {key: loaded[key].copy() for key in loaded.files}
    count = len(windows)
    if int(raw["delta_phi"].shape[0]) != count:
        raise ValueError("labels.npz row count does not match windows.jsonl.")

    primitive_time = raw["primitive_time"].astype(np.float32)
    delta_phi = raw["delta_phi"].astype(np.float32)
    phi_t = raw.get("phi_t", primitive_time).astype(np.float32)
    phi_future = raw.get("phi_future", phi_t + delta_phi).astype(np.float32)
    delta_phi_raw = raw.get("delta_phi_raw", phi_future - phi_t).astype(np.float32)
    labels = {
        **raw,
        "primitive_time": primitive_time,
        "delta_phi": delta_phi,
        "phi_t": phi_t,
        "phi_future": phi_future,
        "delta_phi_raw": delta_phi_raw,
    }
    consistency_error = np.abs((phi_future - phi_t) - delta_phi_raw)
    legacy_gap = delta_phi_raw - delta_phi
    monotonic_rate, monotonic_count = _monotonicity_violations(windows, phi_t, eps=float(eps))

    metrics: dict[str, float] = {
        "num_windows": float(count),
        "has_phi_t": float("phi_t" in raw),
        "has_phi_future": float("phi_future" in raw),
        "has_delta_phi_raw": float("delta_phi_raw" in raw),
        "phi_consistency_abs_error_mean": _finite_mean(consistency_error),
        "phi_consistency_abs_error_max": float(np.max(consistency_error)) if consistency_error.size else 0.0,
        "legacy_delta_gap_mean": _finite_mean(legacy_gap),
        "legacy_delta_gap_abs_mean": _finite_mean(np.abs(legacy_gap)),
        "phi_t_monotonic_violation_rate": monotonic_rate,
        "phi_t_monotonic_violation_count": float(monotonic_count),
    }
    metrics.update(_series_stats("phi_t", phi_t, eps=float(eps)))
    metrics.update(_series_stats("phi_future", phi_future, eps=float(eps)))
    metrics.update(_series_stats("delta_phi_raw", delta_phi_raw, eps=float(eps)))
    metrics.update(_series_stats("legacy_delta_phi", delta_phi, eps=float(eps)))

    out_dir = Path(output_dir) if output_dir is not None else windows_path / "potential_gain_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    rows = _group_rows(windows, labels, eps=float(eps))
    if rows:
        with (out_dir / "by_split_source_stage.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    index = _load_index(windows_path)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "windows_dir": str(windows_path),
                "eps": float(eps),
                "index_summary": {
                    "num_windows": index.get("num_windows"),
                    "source_to_id": index.get("source_to_id"),
                    "stage_to_id": index.get("stage_to_id"),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit explicit Phi/Phi-future/DeltaPhi labels in prepared windows.")
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--eps", type=float, default=1.0e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir if args.output_dir else None
    metrics = audit_potential_gain(args.windows_dir, output_dir=output_dir, eps=float(args.eps))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
