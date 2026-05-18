import math

import pytest


torch = pytest.importorskip("torch")


def _masked_pctl(values: torch.Tensor | None, mask: torch.Tensor, q: float) -> float:
    if values is None:
        return float("nan")
    if mask.dtype is not torch.bool:
        mask = mask.bool()
    if not mask.any():
        return float("nan")
    masked = values[mask]
    if masked.numel() == 0:
        return float("nan")
    masked = masked.float()
    masked = masked[torch.isfinite(masked)]
    if masked.numel() == 0:
        return float("nan")
    return float(torch.quantile(masked, q).item())


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.int64,
    ],
)
def test_masked_pctl_handles_dtypes(dtype):
    values = torch.tensor([0, 1, 2, 3, 4], dtype=dtype)
    mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
    p90 = _masked_pctl(values, mask, 0.9)
    assert math.isfinite(p90)


def test_masked_pctl_empty_mask():
    values = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([0, 0, 0], dtype=torch.bool)
    p90 = _masked_pctl(values, mask, 0.9)
    assert math.isnan(p90)


def test_masked_pctl_filters_nonfinite():
    values = torch.tensor([float("nan"), float("inf"), 3.0, 4.0])
    mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    p50 = _masked_pctl(values, mask, 0.5)
    assert math.isfinite(p50)
