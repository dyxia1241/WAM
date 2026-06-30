from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def _to_numpy(values: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def compute_metrics(
    pred_delta_phi: np.ndarray | torch.Tensor | Iterable[float],
    target_delta_phi: np.ndarray | torch.Tensor | Iterable[float],
    pos_delta_phi: np.ndarray | torch.Tensor | Iterable[float] | None = None,
    neg_delta_phi: np.ndarray | torch.Tensor | Iterable[float] | None = None,
) -> dict[str, float]:
    pred = _to_numpy(pred_delta_phi)
    target = _to_numpy(target_delta_phi)
    if pred.shape != target.shape:
        raise ValueError("pred_delta_phi and target_delta_phi must have the same shape.")

    metrics = {
        "delta_phi_mae": float(np.mean(np.abs(pred - target))),
        "delta_phi_rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
    }

    if pos_delta_phi is not None and neg_delta_phi is not None:
        pos = _to_numpy(pos_delta_phi)
        neg = _to_numpy(neg_delta_phi)
        if pos.shape != neg.shape:
            raise ValueError("pos_delta_phi and neg_delta_phi must have the same shape.")
        margin = pos - neg
        metrics.update(
            {
                "ranking_acc": float(tie_aware_ranking(pos, neg)),
                "mean_margin": float(np.mean(margin)),
                "margin_std": float(np.std(margin)),
            }
        )
    return metrics


def summarize_by_type(
    pos_delta_phi: np.ndarray | torch.Tensor,
    neg_delta_phi: np.ndarray | torch.Tensor,
    negative_types: Iterable[str],
) -> dict[str, float]:
    pos = _to_numpy(pos_delta_phi)
    neg = _to_numpy(neg_delta_phi)
    types = np.asarray(list(negative_types))
    if pos.shape != neg.shape or pos.shape[0] != types.shape[0]:
        raise ValueError("pos, neg, and negative_types must have matching lengths.")

    summary: dict[str, float] = {}
    for negative_type in sorted(set(types.tolist())):
        mask = types == negative_type
        summary[f"{negative_type}_ranking_acc"] = float(tie_aware_ranking(pos[mask], neg[mask]))
        summary[f"{negative_type}_mean_margin"] = float(np.mean(pos[mask] - neg[mask]))
    return summary


def tie_aware_ranking(
    pos_delta_phi: np.ndarray | torch.Tensor | Iterable[float],
    neg_delta_phi: np.ndarray | torch.Tensor | Iterable[float],
) -> float:
    pos = _to_numpy(pos_delta_phi)
    neg = _to_numpy(neg_delta_phi)
    if pos.shape != neg.shape:
        raise ValueError("pos_delta_phi and neg_delta_phi must have the same shape.")
    wins = pos > neg
    ties = np.isclose(pos, neg, rtol=1e-7, atol=1e-9)
    return float(np.mean(wins.astype(np.float64) + 0.5 * ties.astype(np.float64)))
