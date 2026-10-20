"""Host DSP (numpy) against the training-time torch implementation, when torch and the research code
are available (skipped in a plain install)."""
import os, sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch = pytest.importorskip("torch")
if not os.path.exists(os.path.join(ROOT, "training", "tiny_model.py")):
    pytest.skip("research code not present", allow_module_level=True)
sys.path.insert(0, os.path.join(ROOT, "training"))
import tiny_model  # noqa: E402
from hibari import dsp  # noqa: E402


def test_synth_matches_torch_to_wav():
    rng = np.random.default_rng(0)
    F, bins = 120, 601
    parts = [rng.normal(-3, 1, (bins, F)).astype(np.float32) for _ in range(5)]
    parts[1] = rng.normal(0, 1, (bins, F)).astype(np.float32)          # theta
    parts[4] = rng.uniform(-np.pi, np.pi, (bins, F)).astype(np.float32)  # phi
    f0 = np.where(np.arange(F) % 40 < 25, 180.0, 0.0).astype(np.float32)

    class ZeroNoise:
        def standard_normal(self, shape, dtype=np.float32):
            return np.zeros(shape, dtype)
    ours = dsp.synth(*parts, f0, ZeroNoise())
    tp = [torch.from_numpy(p)[None] for p in parts]
    torch.manual_seed(0)
    lh, th, ln, lr, phi = tp
    ref = tiny_model.to_wav((lh, th, torch.full_like(ln, -40.0), lr, phi), torch.from_numpy(f0)[None])[0].numpy()
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())
