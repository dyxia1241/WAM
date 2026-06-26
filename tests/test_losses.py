import torch
import pytest

from mvp0.losses import counterfactual_ranking_loss, delta_phi_loss, ranking_accuracy


def test_counterfactual_loss_smaller_when_positive_is_higher():
    pos_good = torch.tensor([[3.0], [2.0]])
    neg_good = torch.tensor([[-1.0], [-2.0]])
    pos_bad = torch.tensor([[-1.0], [-2.0]])
    neg_bad = torch.tensor([[3.0], [2.0]])

    good_loss = counterfactual_ranking_loss(pos_good, neg_good)
    bad_loss = counterfactual_ranking_loss(pos_bad, neg_bad)

    assert good_loss < bad_loss


def test_ranking_accuracy():
    pos = torch.tensor([[3.0], [-1.0], [2.0]])
    neg = torch.tensor([[1.0], [2.0], [1.0]])

    acc = ranking_accuracy(pos, neg)

    assert acc.item() == pytest.approx(2 / 3)


def test_delta_phi_loss_is_finite():
    logits = torch.tensor([[0.0], [2.0]])
    target = torch.tensor([0.5, 0.9])

    loss = delta_phi_loss(logits, target)

    assert torch.isfinite(loss)
