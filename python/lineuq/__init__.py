"""LineUQ -- measured line-segment localisation uncertainty.

A downstream geometric estimator is usually handed endpoints and nothing else,
so it treats a segment as a unit vote or weights it by length.  A segment fitted
by total least squares already carries the statistics that say how well it was
fitted -- the sample count ``N`` and the two eigenvalues of their covariance --
and this package measures them for any detector's output, from the image and the
endpoints alone::

    import lineuq

    st = lineuq.measure(image, segments)     # image (H, W) uint8, segments (N, 4)
    w = st.sqrt_fisher()                     # a per-segment weight, no tuning

If your detector already accumulates its own moments, you do not need an image
at all -- :func:`weight_pow` turns the three statistics into the same weight, so
a native supplier and the posterior pass are interchangeable.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from ._core import EstimatorParams, Params, WEIGHT_CAP

__all__ = [
    "Stats",
    "Frame",
    "Manhattan",
    "Params",
    "EstimatorParams",
    "WEIGHT_CAP",
    "measure",
    "weight_pow",
    "sqrt_fisher",
    "estimate_manhattan",
    "__version__",
]

__version__ = "0.1.0"


@dataclass(frozen=True)
class Stats:
    """Per-segment measurements, one numpy column per quantity.

    ``lambda_max`` and ``lambda_min`` are eigenvalues of the *normalised*
    covariance of the accepted ridge samples, so both are variances in px^2: a
    segment of length ``l`` whose samples scatter about the fitted line with
    standard deviation ``sigma_e`` has ``lambda_max ~ l**2 / 12`` and
    ``lambda_min ~ sigma_e**2``, whatever the sample count.

    ``fallback`` marks segments the pass could not measure.  Those report an
    isotropic scatter, which is the *lowest* weight in the family rather than an
    accidentally large one -- a per-segment statement a support set sized by
    construction cannot make.
    """

    n: np.ndarray
    lambda_max: np.ndarray
    lambda_min: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    dir_x: np.ndarray
    dir_y: np.ndarray
    sections_total: np.ndarray
    sections_used: np.ndarray
    fallback: np.ndarray

    def __len__(self) -> int:
        return int(self.n.shape[0])

    @property
    def sigma2(self) -> np.ndarray:
        """Plug-in direction-variance proxy, ``lambda_min / (n * lambda_max)``.

        Exactly zero where the accepted samples were collinear, which is a real
        answer rather than a failure: nothing about that fit was uncertain.  The
        weight accessors saturate on it instead of dividing by it.
        """
        out = np.zeros_like(self.n)
        ok = (self.n > 0) & (self.lambda_max > 0)
        out[ok] = self.lambda_min[ok] / (self.n[ok] * self.lambda_max[ok])
        return out

    def weight_pow(self, q: float = 1.0, cap: float = WEIGHT_CAP) -> np.ndarray:
        """The family ``(n * lambda_max / lambda_min) ** (q / 2)``, clamped.

        ``q = 1`` is :meth:`sqrt_fisher`, ``q = 2`` the untempered
        inverse-variance weight.  Which exponent is best is regime-dependent,
        which is why this package hands back the statistics and not only one
        weight.
        """
        return _core.weight_pow(
            np.ascontiguousarray(self.n, dtype=np.float64),
            np.ascontiguousarray(self.lambda_max, dtype=np.float64),
            np.ascontiguousarray(self.lambda_min, dtype=np.float64),
            float(q),
            float(cap),
        )

    def sqrt_fisher(self, cap: float = WEIGHT_CAP) -> np.ndarray:
        """The parameter-free default weight, ``sqrt(n * lambda_max / lambda_min)``."""
        return self.weight_pow(1.0, cap)

    @classmethod
    def from_moments(cls, n, lambda_max, lambda_min) -> "Stats":
        """Build a :class:`Stats` from moments some other code accumulated.

        For a detector that already fits its lines to samples of the ridge, this
        is the whole integration: it has the three numbers, and everything
        downstream in this package works on them without an image.
        """
        n = np.ascontiguousarray(n, dtype=np.float64).ravel()
        a = np.ascontiguousarray(lambda_max, dtype=np.float64).ravel()
        b = np.ascontiguousarray(lambda_min, dtype=np.float64).ravel()
        if not (n.shape == a.shape == b.shape):
            raise ValueError("n, lambda_max and lambda_min must be the same length")
        z = np.zeros_like(n)
        return cls(
            n=n,
            lambda_max=a,
            lambda_min=b,
            cx=z.copy(),
            cy=z.copy(),
            dir_x=np.ones_like(n),
            dir_y=z.copy(),
            sections_total=np.zeros_like(n, dtype=np.int64),
            sections_used=np.zeros_like(n, dtype=np.int64),
            fallback=np.zeros_like(n, dtype=bool),
        )


def _wrap(d: dict) -> Stats:
    return Stats(**d)


def _as_segments(segments) -> np.ndarray:
    a = np.ascontiguousarray(segments, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("segments must have shape (N, 4): x0 y0 x1 y1")
    return a


class Frame:
    """The per-image half of the measurement: one blur and one gradient pass.

    Build one per image and reuse it across every segment you want measured on
    that image; walking a segment is cheap next to preparing the frame.
    """

    def __init__(self, image):
        self._f = _core.Frame(np.ascontiguousarray(image, dtype=np.uint8))

    @property
    def width(self) -> int:
        return self._f.width

    @property
    def height(self) -> int:
        return self._f.height

    @property
    def valid(self) -> bool:
        return self._f.valid

    def measure(self, segments, params: Params | None = None) -> Stats:
        return _wrap(
            self._f.measure(_as_segments(segments), params or Params())
        )


def measure(image, segments, params: Params | None = None) -> Stats:
    """Measure every segment against one image.  See :class:`Stats`."""
    return _wrap(
        _core.measure(
            np.ascontiguousarray(image, dtype=np.uint8),
            _as_segments(segments),
            params or Params(),
        )
    )


def weight_pow(n, lambda_max, lambda_min, q: float = 1.0, cap: float = WEIGHT_CAP):
    """Weights straight from three moment columns, with no image involved.

    This is the interface a detector that supplies its own moments should use.
    """
    return _core.weight_pow(
        np.ascontiguousarray(n, dtype=np.float64).ravel(),
        np.ascontiguousarray(lambda_max, dtype=np.float64).ravel(),
        np.ascontiguousarray(lambda_min, dtype=np.float64).ravel(),
        float(q),
        float(cap),
    )


def sqrt_fisher(n, lambda_max, lambda_min, cap: float = WEIGHT_CAP):
    """``weight_pow(..., q=1)`` -- the parameter-free default."""
    return weight_pow(n, lambda_max, lambda_min, 1.0, cap)


@dataclass(frozen=True)
class Manhattan:
    """An estimated orthogonal frame: three unit vanishing directions."""

    ok: bool
    axes: np.ndarray      # (3, 3), one direction per row
    support: float
    lines_used: int


def estimate_manhattan(
    segments,
    camera,
    weights=None,
    min_length: float = 0.0,
    params: EstimatorParams | None = None,
) -> Manhattan:
    """Fit an orthogonal Manhattan frame from calibrated segments.

    ``camera`` is ``(fx, fy, cx, cy)``.  ``weights`` may be ``None`` (unit
    votes), segment lengths, ``Stats.sqrt_fisher()``, or anything else: the
    weight enters in exactly one place, the refit of each axis, so swapping it
    is a controlled comparison rather than three different estimators.
    """
    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()
    d = _core.estimate_manhattan(
        _as_segments(segments),
        tuple(float(v) for v in camera),
        weights,
        float(min_length),
        params or EstimatorParams(),
    )
    return Manhattan(
        ok=bool(d["ok"]),
        axes=d["axes"],
        support=float(d["support"]),
        lines_used=int(d["lines_used"]),
    )
