from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mvp0.gm100_signal_intervals import (
    WAMBoundaryCandidate,
    build_local_step_intervals_for_episode,
    dataclass_json,
    interval_to_wam_boundary,
    load_task_annotations,
    load_task_name,
    read_episode_signals,
)
from mvp0.import_gm100 import episode_index, wam_episode_id


LABEL_SOURCE = "gm100_signal_native_local_step_interval_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WAM primitive progress labels from GM-100 signal intervals.")
    parser.add_argument("--raw-root", required=True, help="Raw GM-100 subset root.")
    parser.add_argument("--annotation-csv", required=True, help="GM-100 task type annotation CSV.")
    parser.add_argument("--episodes-root", default=None, help="WAM episode root. If omitted, only summary files are written.")
    parser.add_argument("--output-dir", default=None, help="Directory for gt_summary.json and intervals.jsonl.")
    parser.add_argument("--manifest", default=None, help="Defaults to <raw-root>/gm100_subset_manifest.json when present.")
    parser.add_argument("--stage", default="move")
    parser.add_argument("--min-interval-span", type=int, default=24)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--limit-episodes", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels.json files.")
    return parser


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected JSON object lines in {path}.")
            rows.append(loaded)
    return rows


def selected_episodes(raw_root: str | Path, manifest: str | Path | None = None) -> list[tuple[str, str]]:
    raw_root = Path(raw_root)
    manifest_path = Path(manifest) if manifest is not None else raw_root / "gm100_subset_manifest.json"
    if manifest_path.exists():
        manifest_obj = read_json(manifest_path)
        selected = manifest_obj.get("selected_episodes", [])
        if not isinstance(selected, list):
            raise ValueError(f"selected_episodes must be a list in {manifest_path}.")
        out = [(str(item["task_id"]), str(item["episode_id"])) for item in selected]
        return sorted(out)

    out: list[tuple[str, str]] = []
    for parquet_path in sorted(raw_root.glob("task_*/data/chunk-000/episode_*.parquet")):
        task_id = parquet_path.parents[2].name
        out.append((task_id, parquet_path.stem))
    return out


def parquet_path(raw_root: str | Path, task_id: str, source_episode_id: str) -> Path:
    return Path(raw_root) / task_id / "data" / "chunk-000" / f"{source_episode_id}.parquet"


def load_wam_episode_index(episodes_root: str | Path | None) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    if episodes_root is None:
        return {}
    root = Path(episodes_root)
    if not root.exists():
        raise FileNotFoundError(root)
    out: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for episode_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        task_id = str(meta["task_id"])
        source_episode_id = str(meta.get("source_episode_id", ""))
        if not source_episode_id:
            continue
        out[(task_id, source_episode_id)] = (episode_dir, meta)
    return out


def default_output_dir(raw_root: Path, episodes_root: Path | None) -> Path:
    if episodes_root is not None:
        return episodes_root / "progress_gt"
    return raw_root / "wam_progress_gt"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def labels_payload(
    task_id: str,
    source_episode_id: str,
    episode_meta: dict[str, Any],
    candidates: list[WAMBoundaryCandidate],
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "primitive_boundaries": [
            {"stage": candidate.stage, "start": int(candidate.start), "end": int(candidate.end)}
            for candidate in candidates
        ],
        "success": bool(episode_meta.get("success", True)),
        "label_source": LABEL_SOURCE,
        "source_task_id": task_id,
        "source_episode_id": source_episode_id,
        "params": params,
        "primitive_metadata": [
            {
                "stage": candidate.stage,
                "start": int(candidate.start),
                "end": int(candidate.end),
                "feasible_windows": int(candidate.feasible_windows),
                "interval": dataclass_json(candidate.interval),
            }
            for candidate in candidates
        ],
    }


def filter_candidates(
    candidates: list[WAMBoundaryCandidate],
    min_interval_span: int,
) -> tuple[list[WAMBoundaryCandidate], list[tuple[WAMBoundaryCandidate, str]]]:
    kept: list[WAMBoundaryCandidate] = []
    dropped: list[tuple[WAMBoundaryCandidate, str]] = []
    previous_end = -1
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        if candidate.interval.span < min_interval_span:
            dropped.append((candidate, "short_interval"))
            continue
        if candidate.feasible_windows <= 0:
            dropped.append((candidate, "no_feasible_windows"))
            continue
        if candidate.start <= previous_end:
            dropped.append((candidate, "overlap"))
            continue
        kept.append(candidate)
        previous_end = candidate.end
    return kept, dropped


def build_progress_gt(
    raw_root: str | Path,
    annotation_csv: str | Path,
    episodes_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    manifest: str | Path | None = None,
    stage: str = "move",
    min_interval_span: int = 24,
    history: int = 4,
    horizon: int = 8,
    stride: int = 2,
    limit_episodes: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    episodes_root_path = Path(episodes_root) if episodes_root is not None else None
    output_dir_path = Path(output_dir) if output_dir is not None else default_output_dir(raw_root, episodes_root_path)
    annotations = load_task_annotations(annotation_csv)
    wam_index = load_wam_episode_index(episodes_root_path)
    selections = selected_episodes(raw_root, manifest)
    if limit_episodes > 0:
        selections = selections[:limit_episodes]

    params = {
        "stage": stage,
        "min_interval_span": min_interval_span,
        "history": history,
        "horizon": horizon,
        "stride": stride,
        "exclude_cross_boundary": True,
        "label_source": LABEL_SOURCE,
    }
    intervals_rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    task_names: dict[str, str] = {}
    summary_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    for task_id, source_episode_id in selections:
        summary_counts["episodes_seen"] += 1
        episode_key = f"{task_id}/{source_episode_id}"
        if task_id not in annotations:
            skip_reasons["missing_annotation"] += 1
            errors.append({"episode": episode_key, "error": "missing_annotation"})
            continue
        raw_episode_path = parquet_path(raw_root, task_id, source_episode_id)
        if not raw_episode_path.exists():
            skip_reasons["missing_parquet"] += 1
            errors.append({"episode": episode_key, "error": "missing_parquet"})
            continue

        try:
            signals = read_episode_signals(raw_episode_path)
            task_names.setdefault(task_id, load_task_name(raw_root, task_id))
            intervals = build_local_step_intervals_for_episode(
                task_id=task_id,
                episode_id=episode_index(source_episode_id),
                signals=signals,
                task_meta=annotations[task_id],
                task_name_raw=task_names[task_id],
            )
        except Exception as exc:  # noqa: BLE001 - summary should keep batch processing alive.
            skip_reasons["exception"] += 1
            errors.append({"episode": episode_key, "error": repr(exc)})
            continue

        if intervals:
            summary_counts["episodes_with_candidate_intervals"] += 1
        else:
            skip_reasons["no_candidate_intervals"] += 1

        episode_dir: Path | None = None
        episode_meta: dict[str, Any] = {}
        if episodes_root_path is not None:
            indexed = wam_index.get((task_id, source_episode_id))
            if indexed is None:
                skip_reasons["missing_wam_episode"] += 1
            else:
                episode_dir, episode_meta = indexed
        num_frames = int(episode_meta.get("num_frames", signals.num_rows)) if episode_meta else signals.num_rows

        candidates_raw = [
            interval_to_wam_boundary(
                interval=interval,
                num_frames=num_frames,
                stage=stage,
                history=history,
                horizon=horizon,
                stride=stride,
                exclude_cross_boundary=True,
            )
            for interval in intervals
        ]
        candidates = [candidate for candidate in candidates_raw if candidate is not None]
        kept, dropped = filter_candidates(candidates, min_interval_span=min_interval_span)
        summary_counts["intervals_total"] += len(intervals)
        summary_counts["intervals_valid"] += len(kept)
        for _, reason in dropped:
            summary_counts[f"intervals_dropped_{reason}"] += 1

        kept_ids = {candidate.interval.interval_id for candidate in kept}
        dropped_reason = {candidate.interval.interval_id: reason for candidate, reason in dropped}
        for candidate in candidates:
            interval = candidate.interval
            intervals_rows.append(
                {
                    "task_id": task_id,
                    "source_episode_id": source_episode_id,
                    "wam_episode_id": wam_episode_id(task_id, source_episode_id),
                    "interval_id": interval.interval_id,
                    "start_row": int(interval.start_row),
                    "end_row_exclusive": int(interval.end_row),
                    "boundary_start": int(candidate.start),
                    "boundary_end": int(candidate.end),
                    "span": int(interval.span),
                    "feasible_windows": int(candidate.feasible_windows),
                    "kept": interval.interval_id in kept_ids,
                    "drop_reason": dropped_reason.get(interval.interval_id, ""),
                    "label_written": False,
                    "active_arm_pattern": interval.active_arm_pattern,
                    "merge_confidence": interval.merge_confidence,
                    "reason_codes": list(interval.reason_codes),
                    "serial_repetition_risk": interval.serial_repetition_risk,
                    "source_anchor_event_ids": list(interval.source_anchor_event_ids),
                    "source_raw_event_ids": list(interval.source_raw_event_ids),
                }
            )

        if not kept:
            if intervals:
                skip_reasons["no_valid_intervals"] += 1
            continue
        summary_counts["episodes_with_valid_intervals"] += 1
        if episode_dir is None:
            continue

        labels_path = episode_dir / "labels.json"
        if labels_path.exists() and not overwrite:
            raise FileExistsError(f"{labels_path} exists. Pass --overwrite to replace it.")
        write_json(labels_path, labels_payload(task_id, source_episode_id, episode_meta, kept, params))
        summary_counts["labels_written"] += 1
        for row in intervals_rows:
            if row["task_id"] == task_id and row["source_episode_id"] == source_episode_id and row["kept"]:
                row["label_written"] = True

    summary = {
        "raw_root": str(raw_root),
        "episodes_root": str(episodes_root_path) if episodes_root_path is not None else "",
        "annotation_csv": str(annotation_csv),
        "output_dir": str(output_dir_path),
        "params": params,
        "counts": dict(summary_counts),
        "skip_reasons": dict(skip_reasons),
        "tasks_seen": len({task_id for task_id, _ in selections}),
        "errors": errors[:100],
    }
    write_json(output_dir_path / "gt_summary.json", summary)
    write_jsonl(output_dir_path / "intervals.jsonl", intervals_rows)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = build_progress_gt(
        raw_root=args.raw_root,
        annotation_csv=args.annotation_csv,
        episodes_root=args.episodes_root,
        output_dir=args.output_dir,
        manifest=args.manifest,
        stage=args.stage,
        min_interval_span=args.min_interval_span,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        limit_episodes=args.limit_episodes,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(f"wrote GT summary to {summary['output_dir']}")


if __name__ == "__main__":
    main()
