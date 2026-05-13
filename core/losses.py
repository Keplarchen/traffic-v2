"""Pinball (quantile) loss.

The minimizer of this loss is the conditional tau-quantile of the target
distribution (Koenker & Bassett, 1978).
"""

import torch


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus=(0.5, 0.9, 0.95),
) -> torch.Tensor:
    """
    pred:   [..., Q]   any leading dims, Q = len(taus) on the last dim
    target: [...]      same shape as pred without the last dim
    Returns: scalar, mean over all dims.
    """
    target = target.unsqueeze(-1)                                    # [..., 1]
    diff = target - pred                                             # [..., Q]
    taus_t = torch.as_tensor(taus, device=pred.device, dtype=pred.dtype)
    taus_t = taus_t.view(*([1] * (pred.ndim - 1)), -1)               # broadcast to [..., Q]
    loss = torch.maximum(taus_t * diff, (taus_t - 1) * diff)
    return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, P, Q = 8, 462, 3
    pred = torch.randn(B, P, Q, requires_grad=True)
    target = torch.randn(B, P)

    # 1. Basic forward + backward
    loss = pinball_loss(pred, target)
    loss.backward()
    print(f"loss = {loss.item():.4f}, grad finite? {torch.isfinite(pred.grad).all().item()}")

    # 2. Sanity: at tau=0.5, pinball == 0.5 * MAE
    pred50 = torch.randn(B, P, 1, requires_grad=True)
    pinball_50 = pinball_loss(pred50, target, taus=(0.5,))
    mae = (pred50.squeeze(-1) - target).abs().mean()
    ratio = (pinball_50 / mae).item()
    print(f"tau=0.5: pinball / MAE = {ratio:.4f}  (expected 0.5)")

    # 3. Monotonicity: under- vs over-prediction at tau=0.95
    target_const = torch.zeros(1, 1)
    pred_under = torch.tensor([[[-1.0]]])
    pred_over = torch.tensor([[[1.0]]])
    l_under = pinball_loss(pred_under, target_const, taus=(0.95,)).item()
    l_over = pinball_loss(pred_over, target_const, taus=(0.95,)).item()
    print(f"tau=0.95: under loss={l_under:.3f}, over loss={l_over:.3f}, ratio={l_under/l_over:.1f} (expected 19)")

    # 4. Multi-horizon (4D input) shape test
    B, P, H, Q = 4, 462, 3, 3
    pred_mh = torch.randn(B, P, H, Q, requires_grad=True)
    target_mh = torch.randn(B, P, H)
    loss_mh = pinball_loss(pred_mh, target_mh)
    loss_mh.backward()
    print(f"multi-horizon: pred {tuple(pred_mh.shape)}, target {tuple(target_mh.shape)}, "
          f"loss={loss_mh.item():.4f}, grad finite? {torch.isfinite(pred_mh.grad).all().item()}")
