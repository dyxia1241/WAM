from __future__ import annotations

import torch
import torch.nn.functional as F


def delta_phi_loss(delta_phi_logit: torch.Tensor, target_delta_phi: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(delta_phi_logit).reshape_as(target_delta_phi.float())
    return F.smooth_l1_loss(pred, target_delta_phi.float())


def counterfactual_ranking_loss(
    pos_delta_phi_logit: torch.Tensor,
    neg_delta_phi_logit: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    pos = torch.sigmoid(pos_delta_phi_logit)
    neg = torch.sigmoid(neg_delta_phi_logit)
    return -F.logsigmoid(pos - neg - margin).mean()


def ranking_accuracy(pos_delta_phi_logit: torch.Tensor, neg_delta_phi_logit: torch.Tensor) -> torch.Tensor:
    pos = torch.sigmoid(pos_delta_phi_logit)
    neg = torch.sigmoid(neg_delta_phi_logit)
    return (pos > neg).float().mean()

