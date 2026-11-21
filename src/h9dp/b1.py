"""B1: a linear thermal trend plus shallow boosted oblivious decision trees."""

from __future__ import annotations

import torch


def ridge_trend(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
                scale: torch.Tensor, penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-scaled primal slope via a sample-space solve, with unpenalized intercept."""
    prob = weight / weight.sum()
    mean = prob @ x
    target_mean = prob @ y
    root = weight.sqrt()
    design = (x - mean) * scale * root[:, None]
    gram = design @ design.T + penalty * torch.eye(len(x), dtype=x.dtype, device=x.device)
    dual = torch.linalg.solve(gram, root * (y - target_mean))
    slope = (design.T @ dual) * scale
    return slope, target_mean - mean @ slope


def grow_tree(gates: torch.Tensor, error: torch.Tensor, weight: torch.Tensor,
              depth: int, shrinkage: float, penalty: float):
    """One level-wise greedy tree. Every node at a level uses the same split.

    Each candidate is a binary feature-threshold comparison on the *training*
    records. Score only splits with support in both children. The resulting
    leaf corrections shrink towards zero, including for unpopulated leaves.
    """
    node = torch.zeros(len(error), dtype=torch.long, device=error.device)
    chosen = []
    for level in range(depth):
        membership = node[:, None] == torch.arange(2 ** level, device=node.device)
        mass = membership * weight[:, None]
        value = mass * error[:, None]
        right, rsum = gates @ mass, gates @ value
        total, sums = mass.sum(0), value.sum(0)
        left, lsum = (total - right).clamp_min(0), sums - rsum
        gain = rsum.square() / (right + penalty) + lsum.square() / (left + penalty) - sums.square() / (total + penalty)
        eligible = (left >= 4) & (right >= 4)
        # An unhelpful level must not split unsupported records merely because
        # all gains tie. A final constant-false candidate provides a no-op.
        score = torch.where(eligible, gain, 0.0).sum(1)
        best = torch.where(score.max() > 0, score.argmax(), len(gates) - 1)
        chosen.append(best)
        bit = gates.index_select(0, best.reshape(1)).squeeze(0).long()
        node = 2 * node + bit
    members = (node[:, None] == torch.arange(2 ** depth, device=node.device)) * weight[:, None]
    leaves = shrinkage * (members.T @ error) / (members.sum(0) + penalty)
    return torch.stack(chosen), leaves, leaves[node]
