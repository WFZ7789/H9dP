"""Transfer cross-fitted pressure residuals between visually similar scenes."""

import torch


def double_deletion_residual(influence, inverse, dual, predicted, target):
    """Residual at each neighbor j after omitting query i and neighbor j."""
    diagonal = inverse.diag()
    ratio = inverse / diagonal[:, None]
    scaled = dual / diagonal
    conditional_prediction = predicted[None, :] - scaled[:, None] * influence.T
    diagonal_after_deletion = diagonal[None, :] - inverse * ratio
    dual_after_deletion = dual[None, :] - scaled[:, None] * inverse
    query_leverage = influence.diag()[None, :] - influence.T * ratio
    residual = (target[None, :] - conditional_prediction
                + query_leverage * dual_after_deletion / diagonal_after_deletion.clamp_min(1e-8))
    same = torch.eye(len(dual), dtype=torch.bool, device=dual.device)
    return residual.masked_fill(same, 0)


def transfer_weights(similarity, settings, *, leave_out=False):
    bandwidth = 0.016 + 0.076 * settings[1].sigmoid()
    strength = 1.6 * settings[0].sigmoid()
    weights = ((similarity.clamp(max=1) - 0.82) / bandwidth).exp()
    if leave_out:
        same = torch.eye(len(weights), dtype=torch.bool, device=weights.device)
        weights = weights.masked_fill(same, 0)
    return strength * weights / (1.0 + weights.sum(dim=1, keepdim=True))
