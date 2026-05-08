"""Tests for ``src.tse.train._build_metrics_fn``.

Specifically validates the active-channel masking introduced to ignore
zero-padded GT channels (i.e. the case where ``n_fg < num_output_channels``).
"""

from __future__ import annotations

import pytest
import torch
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio as si_sdr,
    signal_noise_ratio as snr_fn,
)

from src.tse.train import _build_metrics_fn


def _make_batch(num_fg_per_sample: list[int], C: int = 5, T: int = 1024, sr: int = 16000):
    """Build (outputs, targets, inputs) where active channels are signal,
    padded channels are exact zeros (matching SoundscapeDataset behavior)."""
    B = len(num_fg_per_sample)
    torch.manual_seed(0)

    # Random GT, then zero out padded channels
    gt = torch.randn(B, C, T)
    for b, n in enumerate(num_fg_per_sample):
        gt[b, n:] = 0.0

    # est = gt + small noise → high si_sdr on active channels
    est = gt + 0.01 * torch.randn(B, C, T)
    # Padded channels: keep est = 0 to mimic ill-defined si_sdr region
    for b, n in enumerate(num_fg_per_sample):
        est[b, n:] = 0.0

    mix = torch.randn(B, 2, T)  # (B, M=2, T) binaural mixture

    outputs = {"output": est}
    targets = {
        "target": gt,
        "num_fg_labels": torch.tensor(num_fg_per_sample, dtype=torch.long),
    }
    inputs = {"mixture": mix}
    return outputs, targets, inputs


def _reference_metric(outputs, targets, inputs):
    """Per-sample reference: slice [0:n_fg] explicitly, average across active rows."""
    est = outputs["output"]
    gt = targets["target"]
    mix = inputs["mixture"]
    num_fg = targets["num_fg_labels"]

    B, C, T = est.shape
    mix_mono = mix.mean(dim=1, keepdim=True).expand(B, C, T)

    si_rows, snr_rows = [], []
    for b in range(B):
        n = int(num_fg[b])
        if n == 0:
            continue
        e, g, m = est[b, :n], gt[b, :n], mix_mono[b, :n]
        si_rows.append(si_sdr(e, g) - si_sdr(m, g))  # (n,)
        snr_rows.append(snr_fn(e, g) - snr_fn(m, g))  # (n,)
    if not si_rows:
        return {"si_sdri": torch.zeros(()), "snri": torch.zeros(())}
    return {
        "si_sdri": torch.cat(si_rows).mean(),
        "snri": torch.cat(snr_rows).mean(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_padded_channels_excluded():
    """Primary case: n_fg < C → vectorised mask must match per-sample reference."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[1, 3, 2], C=5)

    got = metrics_fn(outputs, targets, inputs)
    ref = _reference_metric(outputs, targets, inputs)

    assert torch.allclose(got["si_sdri"], ref["si_sdri"], atol=1e-5)
    assert torch.allclose(got["snri"], ref["snri"], atol=1e-5)


def test_naive_mean_differs_from_masked():
    """Sanity: confirm padded-zero pollution actually changes the result —
    otherwise the mask wouldn't matter."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[1, 1], C=5)

    masked = metrics_fn(outputs, targets, inputs)["si_sdri"]

    # Naive (all-channel) si_sdri — what the buggy version would compute
    est = outputs["output"]
    gt = targets["target"]
    mix = inputs["mixture"]
    mix_mono = mix.mean(dim=1, keepdim=True).expand_as(gt)
    naive = (si_sdr(est, gt) - si_sdr(mix_mono, gt)).mean()

    # Active channels are clean signal (high si_sdri), padded are zero/zero
    # → naive mean is materially different from masked.
    assert not torch.allclose(masked, naive, atol=0.5), (
        f"masked={masked.item()} naive={naive.item()} "
        "— if these match, the test setup didn't trigger the padding case"
    )


def test_n_fg_equals_C():
    """Edge case: every channel active → mask = all-True → equivalent to
    plain mean over (B, C)."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[5, 5], C=5)

    got = metrics_fn(outputs, targets, inputs)

    est = outputs["output"]
    gt = targets["target"]
    mix = inputs["mixture"]
    mix_mono = mix.mean(dim=1, keepdim=True).expand_as(gt)
    expected_si = (si_sdr(est, gt) - si_sdr(mix_mono, gt)).mean()

    assert torch.allclose(got["si_sdri"], expected_si, atol=1e-5)


def test_all_zero_n_fg():
    """Edge case: no active channels in batch → return zero tensors, not NaN."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[0, 0], C=5)

    got = metrics_fn(outputs, targets, inputs)
    assert torch.equal(got["si_sdri"], torch.zeros(()))
    assert torch.equal(got["snri"], torch.zeros(()))


def test_mixed_zero_and_active():
    """One sample inactive (n_fg=0), rest active — the zero sample is dropped
    by the mask, the rest still produce valid metrics."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[0, 2, 1], C=5)

    got = metrics_fn(outputs, targets, inputs)
    ref = _reference_metric(outputs, targets, inputs)

    assert torch.allclose(got["si_sdri"], ref["si_sdri"], atol=1e-5)
    assert torch.allclose(got["snri"], ref["snri"], atol=1e-5)


def test_device_consistency_cpu():
    """num_fg_labels is moved onto est.device internally — passing CPU
    long tensor with CPU est must just work."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[1, 2], C=5)
    # Force num_fg_labels onto a tensor that's a different dtype variant
    targets["num_fg_labels"] = targets["num_fg_labels"].to(torch.int32)
    got = metrics_fn(outputs, targets, inputs)
    assert "si_sdri" in got and "snri" in got
    assert torch.isfinite(got["si_sdri"]) and torch.isfinite(got["snri"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_consistency_cuda():
    """End-to-end on CUDA: tensors on GPU, num_fg_labels on CPU
    (mirrors real DataLoader/trainer behaviour) — adapter must move it."""
    metrics_fn = _build_metrics_fn()
    outputs, targets, inputs = _make_batch(num_fg_per_sample=[1, 3], C=5)
    outputs = {k: v.cuda() for k, v in outputs.items()}
    targets["target"] = targets["target"].cuda()
    inputs = {k: v.cuda() for k, v in inputs.items()}
    # num_fg_labels intentionally left on CPU
    got = metrics_fn(outputs, targets, inputs)
    assert got["si_sdri"].device.type == "cuda"
    assert got["snri"].device.type == "cuda"
