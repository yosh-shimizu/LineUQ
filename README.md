# LineUQ

**Measured line-segment localisation uncertainty, and geometric estimation that consumes it.**

Already have a line detector? Add a measured per-segment localisation uncertainty
without modifying it.

```cpp
lineuq::Frame frame(image);                 // one blur + gradient per image
lineuq::Stats st = frame.measure(segment);  // walk the ridge under the endpoints

st.n;  st.lambda_max;  st.lambda_min;       // the three statistics
st.sigma2();                                // direction variance of the fit
st.sqrtFisher();                            // a per-segment weight, no tuning
```

Downstream estimators mostly treat a detected segment as a unit vote, or weight it
by length. A segment fitted by total least squares already carries the statistics
that say how well it was fitted — the number of contributing samples `N` and the
two eigenvalues of their covariance — so `I = N·λmax/λmin` estimates the
directional information it carries, and `w = √I` is a per-segment weight with no
tuning parameters.

LineUQ measures those statistics from the image and a pair of endpoints, so any
detector's output can carry them.

---

## Why re-measure, instead of using the detector's own support pixels?

Because a support region is not, in general, a measurement of localisation. Its
lateral spread is set by the acceptance rule that drew it — a region width, an
angle tolerance, the connectivity of a drawn chain — so on real imagery it
collapses to a function of length and tells the estimator almost nothing the
endpoints had not already said.

Taking the moments from three detectors' own support pixels instead of
re-measuring the ridge raised the median attitude error by 33–77% in our
experiments, landing *below* unit votes. The intensity ridge has to be measured,
not inherited. That is the whole reason this library exists.

---

## Build

No dependencies. C++17 and CMake:

```bash
cmake -S . -B build
cmake --build build
./build/lineuq_tests           # 44 checks
```

`stb_image.h` is vendored under `third_party/` and is used only by the example
programs, never by the library.

## Use

```cpp
#include "lineuq/lineuq.hpp"

lineuq::ImageView view{pixels, width, height, stride};   // your bytes, not ours
std::vector<lineuq::Segment> segs = /* from any detector: x0 y0 x1 y1 */;

const lineuq::Frame frame(view);
for (const auto& s : segs) {
    const lineuq::Stats st = frame.measure(s);
    if (st.fallback) continue;      // the pass could not measure this one
    use(st.sqrtFisher());
}
```

`Frame` holds the per-image work (a Gaussian blur and a gradient pass); the walk
along each segment is the per-segment cost. Build one `Frame` per image and reuse
it.

### The weight is a family, not a constant

`sqrtFisher()` is the parameter-free default, and the one we would ship if one
value had to ship. It is not universally optimal: the best exponent in
`weightPow(q)` is regime-dependent, and in some of our datasets the untempered
`q = 2` was selected instead. This is why `measure()` returns the statistics and
not only a weight — the exponent is the consumer's to choose.

Weights saturate at `Stats::kWeightCap` (1e6). A perfectly collinear fit carries
infinite information, which no downstream sum can use; the cap resolves that
*upwards*, because such a segment is the best-measured one present.

### Geometric estimation

```cpp
#include "lineuq/estimate.hpp"

lineuq::Camera cam{fx, fy, cx, cy};
auto lines = lineuq::calibrate(segs, cam, weights);   // weights may be empty
auto frame = lineuq::estimateManhattan(lines);        // orthogonal triad = 3 VPs
```

The estimator is deterministic and the weight enters in exactly one place — the
weighted refit of each axis — so swapping `weights` between unit, length and
measured uncertainty is a controlled comparison. `examples/estimate_manhattan.cpp`
runs all three on the same segments and prints the three answers.

## Python

```bash
pip install .        # needs a C++17 compiler; on Windows that means MSVC
```

```python
import numpy as np, lineuq

st = lineuq.measure(image, segments)   # image (H, W) uint8, segments (N, 4)

st.n, st.lambda_max, st.lambda_min     # numpy columns, one entry per segment
st.sigma2                              # direction variance of each fit
st.fallback                            # which segments could not be measured
w = st.sqrt_fisher()                   # the weight, (N,) float64

frame = lineuq.Frame(image)            # reuse across many segment sets
st = frame.measure(segments)

m = lineuq.estimate_manhattan(segments, camera=(fx, fy, cx, cy), weights=w)
m.axes                                 # (3, 3): three orthogonal directions
```

The binding is array-oriented: crossing the boundary per segment would cost more
than the measurement does. Everything takes and returns numpy arrays and the GIL
is released around the work.

**If your detector already accumulates its own moments, you do not need an image
at all.**

```python
w = lineuq.sqrt_fisher(n, lambda_max, lambda_min)      # or Stats.from_moments(...)
```

This is deliberately the same code path the measured route uses, so a native
supplier and the posterior pass produce interchangeable weights. It is the
concrete form of the interface this library argues for.

## Examples

```bash
./build/measure_uncertainty image.png segments.txt --csv stats.csv
./build/estimate_manhattan  image.png segments.txt 500 500 320 240
```

`segments.txt` is one `x0 y0 x1 y1` per line, which most detectors will dump for
you in a few lines of code.

## Relationship to SweepLSD

A detector that fits its lines to samples of the ridge already has these
statistics and can emit them for free.
[SweepLSD](https://github.com/yosh-shimizu/sweeplsd) is the native supplier:
it accumulates the moments inside its streaming labeller, so the three scalars
cost no extra image pass.

```
SweepLSD ─────── native ──────┐
                              ├──► LineUQ estimators
LSD / ELSED / EDLines ─ retrofit ┘
```

LineUQ is the consuming and retrofitting side, and knows nothing about which
detector produced the endpoints it is given.

## Status

v0.1. The measurement, the estimator and the Python bindings are here; the
reproduction harness for the paper's tables is not yet, and will follow.

The measurement was verified against the reference implementation the paper's
results were produced with, on 380 real segments of one EuRoC frame. The MSVC
build — the one the Python extension uses — reproduces it **bit for bit on all
380 segments**. A GCC `-O3` build of the same source agrees to 2·10⁻⁸ relative
with 103 of 380 bit-identical, so what is left is the compiler's floating-point
choices and not the algorithm. Accepted-section counts and the set of
unmeasurable segments match exactly under both.

Four ways of weighting the cross-sections *within* a segment were tried during
the research and none of them helped, so they are deliberately not exposed here.

## Citing

See `CITATION.cff`.

## License

MIT.
