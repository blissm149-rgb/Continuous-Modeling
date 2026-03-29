"""Near-amplitude-null (destructive-interference) detector.

Where multiple scatterers cancel nearly completely, |S| ≈ 0 relative
to the local average.  These cancellation nodes are *maximally sensitive*
to model errors:

  At 10 GHz, a 1 mm position error shifts one contributor's phase by
  2π × 2f₀/c × 0.001 ≈ 0.13 rad, which can turn a perfect null into
  a +10 dB peak — a change no model can predict without knowing the
  exact geometry.

Score = 1 − |S| / local_mean_|S|, clipped to [0, 1].

The local mean is computed over a small (3 × 3 × 5) uniform window so
that the score is relative to the immediate neighbourhood rather than
the global dynamic range.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage


class CancellationDetector:
    """Detect near-null amplitude points in a complex RCS tensor."""

    def __init__(self, window: tuple[int, int, int] = (3, 3, 5)):
        """
        Parameters
        ----------
        window:
            Uniform-filter window size in (az, el, freq) dimensions.
        """
        self._window = window

    def compute(self, tensor: np.ndarray) -> np.ndarray:
        """Return score array shape (N_az, N_el, N_freq) in [0, 1].

        Score is high where |S| is near zero relative to its local average.
        """
        amp = np.abs(tensor)
        local_mean = scipy.ndimage.uniform_filter(
            amp.astype(float),
            size=self._window,
            mode="nearest",
        ) + 1e-30
        near_null = np.clip(1.0 - amp / local_mean, 0.0, 1.0)
        return near_null.astype(float)
