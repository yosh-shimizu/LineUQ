"""Python-side tests. Plain asserts, runnable with pytest or directly.

The C++ tests already cover the measurement's behaviour; these cover the things
only the binding can get wrong -- array shapes and dtypes, that the vectorised
path agrees with the scalar C++ one, that weights can be produced without an
image at all, and that the weight ordering survives the boundary.

SPDX-License-Identifier: MIT
"""

import numpy as np

import lineuq


def step_edge(w, h, px, nx, ny, lo=40, hi=200, soft=2.0, py_=0.0):
    """A soft step edge, the same fixture the C++ tests use."""
    ys, xs = np.mgrid[0:h, 0:w]
    d = (xs - px) * nx + (ys - py_) * ny
    t = np.clip(d / soft + 0.5, 0.0, 1.0)
    return (lo + (hi - lo) * t).astype(np.uint8)


def test_measure_a_straight_edge():
    img = step_edge(200, 200, 100.0, 1.0, 0.0)
    segs = np.array([[100.0, 20.0, 100.0, 180.0]])
    st = lineuq.measure(img, segs)

    assert len(st) == 1
    assert st.n.shape == (1,)
    assert st.n.dtype == np.float64
    assert not st.fallback[0]
    assert st.sections_used[0] > 100
    assert abs(st.lambda_max[0] - 160.0**2 / 12.0) < 250.0
    assert st.lambda_min[0] < 0.05
    assert abs(st.dir_y[0]) > 0.999


def test_frame_is_reusable_and_agrees_with_the_one_shot_form():
    img = step_edge(200, 200, 100.0, 1.0, 0.0)
    segs = np.array([[100.0, 20.0, 100.0, 180.0],
                     [100.0, 40.0, 100.0, 160.0]])

    frame = lineuq.Frame(img)
    assert frame.valid and frame.width == 200 and frame.height == 200

    a = frame.measure(segs)
    b = lineuq.measure(img, segs)
    # Same image, same segments: reusing a frame must not change an answer.
    for name in ("n", "lambda_max", "lambda_min", "cx", "cy"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    # A frame measures any segment set, not just the one it was built with.
    c = frame.measure(segs[:1])
    assert len(c) == 1
    np.testing.assert_array_equal(c.lambda_min, a.lambda_min[:1])


def test_weights_and_their_ordering():
    img = step_edge(200, 200, 100.0, 1.0, 0.0)
    rng = np.random.default_rng(7)
    noisy = np.clip(img.astype(np.float64) + rng.normal(0, 12, img.shape),
                    0, 255).astype(np.uint8)
    seg = np.array([[100.0, 20.0, 100.0, 180.0]])

    clean = lineuq.measure(img, seg)
    dirty = lineuq.measure(noisy, seg)
    flat = lineuq.measure(np.full((200, 200), 128, np.uint8), seg)

    assert dirty.lambda_min[0] > clean.lambda_min[0]
    # measured-well > measured-badly > not measurable at all
    assert clean.sqrt_fisher()[0] > dirty.sqrt_fisher()[0] > flat.sqrt_fisher()[0]
    assert flat.fallback[0]

    # sigma2 and the weight family agree with their definitions where sigma2 > 0
    assert dirty.sigma2[0] > 0
    big = 1e30
    np.testing.assert_allclose(
        dirty.weight_pow(2.0, cap=big)[0], 1.0 / dirty.sigma2[0], rtol=1e-9)
    np.testing.assert_allclose(
        dirty.sqrt_fisher(cap=big)[0], (1.0 / dirty.sigma2[0]) ** 0.5, rtol=1e-9)

    # A perfect fit saturates upwards, never to zero.
    if clean.lambda_min[0] <= 1e-12:
        assert clean.sqrt_fisher()[0] == lineuq.WEIGHT_CAP
        assert clean.sigma2[0] == 0.0


def test_weights_without_an_image():
    """The path a detector that already has its own moments takes."""
    n = np.array([40.0, 40.0])
    lmax = np.array([100.0, 100.0])
    lmin = np.array([0.25, 1.0])          # the second is measured four times worse

    w = lineuq.sqrt_fisher(n, lmax, lmin)
    np.testing.assert_allclose(w, np.sqrt(n * lmax / lmin))
    assert w[0] > w[1]

    # and it is the same function the measured path uses
    st = lineuq.Stats.from_moments(n, lmax, lmin)
    np.testing.assert_array_equal(st.sqrt_fisher(), w)
    np.testing.assert_allclose(st.sigma2, lmin / (n * lmax))

    # degenerate: infinite information, resolved at the cap
    assert lineuq.sqrt_fisher([10.0], [1.0], [0.0])[0] == lineuq.WEIGHT_CAP


def test_params_are_honoured():
    img = step_edge(200, 200, 100.0, 1.0, 0.0)
    seg = np.array([[100.0, 20.0, 100.0, 180.0]])
    # A threshold nothing can pass turns every section into a rejection, and the
    # pass must say so rather than inventing a measurement.
    st = lineuq.measure(img, seg, lineuq.Params(power_threshold=1e12))
    assert st.fallback[0] and st.sections_used[0] == 0


def test_input_validation():
    img = step_edge(64, 64, 32.0, 1.0, 0.0)
    for bad in (np.zeros((3, 3)), np.zeros((2, 5)), np.zeros(4)):
        try:
            lineuq.measure(img, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad segment shape %r was accepted" % (bad.shape,))

    # a colour image is not a greyscale one
    try:
        lineuq.measure(np.zeros((8, 8, 3), np.uint8), np.array([[1.0, 1, 5, 5]]))
    except Exception:
        pass
    else:
        raise AssertionError("a 3-D image was accepted")


def _rotation(a, b, c):
    ca, sa, cb, sb, cc, sc = (np.cos(a), np.sin(a), np.cos(b),
                              np.sin(b), np.cos(c), np.sin(c))
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1.0]])
    ry = np.array([[cb, 0, sb], [0, 1.0, 0], [-sb, 0, cb]])
    rx = np.array([[1.0, 0, 0], [0, ca, -sa], [0, sa, ca]])
    return rz @ ry @ rx


def test_estimate_manhattan():
    cam = (500.0, 500.0, 320.0, 240.0)
    R = _rotation(0.30, -0.22, 0.15)
    rng = np.random.default_rng(11)

    segs = []
    for axis in range(3):
        d = R[:, axis]
        for _ in range(30):
            P = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(4, 8)])
            Q = P + d * rng.uniform(0.6, 1.0)
            if P[2] < 1 or Q[2] < 1:
                continue
            segs.append([cam[0] * P[0] / P[2] + cam[2], cam[1] * P[1] / P[2] + cam[3],
                         cam[0] * Q[0] / Q[2] + cam[2], cam[1] * Q[1] / Q[2] + cam[3]])
    segs = np.array(segs)

    m = lineuq.estimate_manhattan(segs, cam)
    assert m.ok and m.axes.shape == (3, 3) and m.lines_used == len(segs)

    worst = 0.0
    for axis in range(3):
        best = max(abs(float(m.axes[k] @ R[:, axis])) for k in range(3))
        worst = max(worst, np.degrees(np.arccos(min(1.0, best))))
    assert worst < 1.0, worst

    # Deterministic, and the weights reach it: a length weighting must not
    # change which lines enter, only how much each counts.
    m2 = lineuq.estimate_manhattan(segs, cam)
    np.testing.assert_array_equal(m.axes, m2.axes)

    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mw = lineuq.estimate_manhattan(segs, cam, weights=lengths)
    assert mw.ok and mw.lines_used == m.lines_used
    assert mw.support != m.support        # the weight is actually consumed

    # a zero weight drops that line
    w0 = np.ones(len(segs))
    w0[0] = 0.0
    assert lineuq.estimate_manhattan(segs, cam, weights=w0).lines_used == len(segs) - 1

    try:
        lineuq.estimate_manhattan(segs, cam, weights=np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("a mismatched weight vector was accepted")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok   %s" % fn.__name__)
    print("\n%d tests, 0 failed" % len(fns))
