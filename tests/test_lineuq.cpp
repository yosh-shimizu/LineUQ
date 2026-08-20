// LineUQ test harness. No framework: it builds anywhere the library builds.
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "lineuq/estimate.hpp"
#include "lineuq/lineuq.hpp"

namespace {

int g_failed = 0;
int g_checks = 0;

void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

void expectNear(double got, double want, double tol, const std::string& what) {
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        ++g_failed;
        std::printf("  FAIL  %s: got %.6g want %.6g (tol %.3g)\n", what.c_str(), got,
                    want, tol);
    }
}

// ---------------------------------------------------------------------------
// Synthetic imagery
// ---------------------------------------------------------------------------

/// A step edge through (px,py) with unit normal (nx,ny): dark on one side,
/// bright on the other, with `soft` pixels of linear ramp so the ridge has a
/// well-defined sub-pixel position.
std::vector<std::uint8_t> stepEdge(int w, int h, double px, double py, double nx,
                                   double ny, double lo, double hi, double soft) {
    std::vector<std::uint8_t> img(std::size_t(w) * h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double d = (x - px) * nx + (y - py) * ny;
            double t = soft > 0 ? (d / soft + 0.5) : (d > 0 ? 1.0 : 0.0);
            if (t < 0) t = 0;
            if (t > 1) t = 1;
            const double v = lo + (hi - lo) * t;
            img[std::size_t(y) * w + x] = std::uint8_t(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    return img;
}

lineuq::ImageView viewOf(const std::vector<std::uint8_t>& px, int w, int h) {
    lineuq::ImageView v;
    v.data = px.data();
    v.width = w;
    v.height = h;
    v.stride = w;
    return v;
}

// ---------------------------------------------------------------------------
void testMeasuresAStraightEdge() {
    std::printf("measure: a straight edge\n");
    const int W = 200, H = 200;
    const auto px = stepEdge(W, H, 100.0, 0.0, 1.0, 0.0, 40, 200, 2.0);
    const lineuq::Frame frame(viewOf(px, W, H));
    expect(frame.valid(), "frame is valid");

    const lineuq::Segment s{100.0, 20.0, 100.0, 180.0};
    const lineuq::Stats st = frame.measure(s);

    expect(!st.fallback, "a strong edge is measured, not fallen back on");
    expect(st.sections_used > 100, "most cross-sections are admitted");
    // Along the edge the samples spread over ~160 px: l^2/12 = 2133.
    expectNear(st.lambda_max, 160.0 * 160.0 / 12.0, 250.0, "lambda_max ~ l^2/12");
    expect(st.lambda_min < 0.05, "lambda_min is small on a clean edge");
    expect(std::fabs(st.dir_y) > 0.999, "the fitted direction is vertical");
    expect(std::fabs(st.cx - 100.0) < 1.0, "the ridge sits on the edge");

    // The derived quantities agree with their definitions.
    expectNear(st.sigma2(), st.lambda_min / (st.n * st.lambda_max), 1e-12,
               "sigma2 = lambda_min / (N lambda_max)");
    expectNear(st.weightPow(1.0), st.sqrtFisher(), 1e-9, "weightPow(1) = sqrtFisher");
}

void testDegenerateFitSaturatesUpwards() {
    std::printf("measure: a perfect fit saturates upwards\n");
    // A synthetic edge can be localised exactly, so lambda_min lands on zero and
    // the information is infinite.  That is the best-measured segment there is,
    // and the weight must say so -- returning zero would invert the ordering.
    const int W = 200, H = 200;
    const auto px = stepEdge(W, H, 100.0, 0.0, 1.0, 0.0, 40, 200, 2.0);
    const lineuq::Frame frame(viewOf(px, W, H));
    const lineuq::Stats st = frame.measure({100.0, 20.0, 100.0, 180.0});

    if (st.lambda_min <= 1e-12) {
        expect(st.information() > 1e300, "a collinear fit carries infinite information");
        expectNear(st.sqrtFisher(), lineuq::Stats::kWeightCap, 0.0,
                   "and its weight saturates at the cap");
        expectNear(st.weightPow(2.0), lineuq::Stats::kWeightCap, 0.0,
                   "at every exponent");
        expectNear(st.sqrtFisher(25.0), 25.0, 0.0, "the cap is a parameter");
    }

    // Whatever happened above, a measured segment must outweigh an unmeasurable
    // one, and the identity must hold wherever sigma2 is positive.
    const std::vector<std::uint8_t> flat(std::size_t(W) * H, 128);
    const lineuq::Stats none =
        lineuq::Frame(viewOf(flat, W, H)).measure({20.0, 100.0, 180.0, 100.0});
    expect(st.sqrtFisher() > none.sqrtFisher(), "measured outranks unmeasurable");
}

void testDirectionFollowsTheEdge() {
    std::printf("measure: a tilted edge\n");
    const int W = 240, H = 240;
    const double ang = 30.0 * 3.14159265358979 / 180.0;
    // Edge direction (cos,sin); its normal is (-sin,cos).
    const double ux = std::cos(ang), uy = std::sin(ang);
    const auto px = stepEdge(W, H, 120.0, 120.0, -uy, ux, 40, 210, 2.0);
    const lineuq::Frame frame(viewOf(px, W, H));

    const double t = 70.0;
    const lineuq::Segment s{120 - ux * t, 120 - uy * t, 120 + ux * t, 120 + uy * t};
    const lineuq::Stats st = frame.measure(s);
    expect(!st.fallback, "the tilted edge is measured");
    // dir is defined up to sign.
    const double c = std::fabs(st.dir_x * ux + st.dir_y * uy);
    expectNear(c, 1.0, 2e-3, "the fitted direction follows the edge");
}

void testNoiseRaisesLambdaMin() {
    std::printf("measure: noise raises lambda_min\n");
    const int W = 200, H = 200;
    auto clean = stepEdge(W, H, 100.0, 0.0, 1.0, 0.0, 40, 200, 2.0);
    auto noisy = clean;
    std::mt19937 rng(7);
    std::normal_distribution<double> g(0.0, 12.0);
    for (auto& v : noisy) {
        const double n = double(v) + g(rng);
        v = std::uint8_t(n < 0 ? 0 : (n > 255 ? 255 : n));
    }
    const lineuq::Segment s{100.0, 20.0, 100.0, 180.0};
    const lineuq::Stats a = lineuq::Frame(viewOf(clean, W, H)).measure(s);
    const lineuq::Stats b = lineuq::Frame(viewOf(noisy, W, H)).measure(s);
    expect(!a.fallback && !b.fallback, "both are measured");
    expect(b.lambda_min > a.lambda_min,
           "the noisy edge is localised less precisely");
    expect(b.sqrtFisher() < a.sqrtFisher(),
           "and therefore earns a smaller weight");

    // On the noisy edge sigma2 is strictly positive, so the identities that the
    // degenerate case saturates out of hold exactly here.
    expect(b.sigma2() > 0, "the noisy fit has a positive variance");
    expectNear(b.information() * b.sigma2(), 1.0, 1e-9, "I = 1 / sigma2");
    // Raise the cap out of the way first: at q = 2 the untempered weight runs
    // past the default 1e6 on perfectly ordinary data, which is the cap doing
    // its job rather than an edge case.
    const double big = 1e30;
    expectNear(b.weightPow(2.0, big), b.information(), 1e-6 * b.information(),
               "weightPow(2) = I");
    expectNear(b.sqrtFisher(big), std::sqrt(b.information()),
               1e-6 * b.sqrtFisher(big), "sqrtFisher = sqrt(I)");
    expect(b.weightPow(2.0) == lineuq::Stats::kWeightCap,
           "and the default cap clips it");
}

void testFallbackOnNothing() {
    std::printf("measure: the fallback reports failure\n");
    const int W = 120, H = 120;
    const std::vector<std::uint8_t> flat(std::size_t(W) * H, 128);
    const lineuq::Frame frame(viewOf(flat, W, H));
    const lineuq::Stats st = frame.measure({20.0, 60.0, 100.0, 60.0});
    expect(st.fallback, "a flat image cannot be measured");
    expect(st.sections_used == 0, "no cross-section is admitted");
    expectNear(st.lambda_min, st.lambda_max, 1e-9,
               "the fallback scatter is isotropic");

    // Isotropic means the smallest weight in the family, not the largest: a
    // segment we could not measure must not outrank one we could.
    const auto px = stepEdge(W, H, 60.0, 0.0, 1.0, 0.0, 40, 200, 2.0);
    const lineuq::Stats good =
        lineuq::Frame(viewOf(px, W, H)).measure({60.0, 20.0, 60.0, 100.0});
    expect(st.sqrtFisher() < good.sqrtFisher(),
           "an unmeasurable segment weighs less than a measured one");
}

void testEmptyAndDegenerateInputs() {
    std::printf("measure: degenerate inputs\n");
    const lineuq::Frame none{};
    expect(!none.valid(), "a default frame is invalid");
    expect(none.measure({0, 0, 10, 10}).fallback, "and measures nothing");

    const int W = 60, H = 60;
    const auto px = stepEdge(W, H, 30.0, 0.0, 1.0, 0.0, 40, 200, 2.0);
    const lineuq::Frame frame(viewOf(px, W, H));
    expect(frame.measure({30.0, 30.0, 30.0, 30.0}).fallback,
           "a zero-length segment falls back");

    const std::vector<lineuq::Stats> many =
        lineuq::measure(viewOf(px, W, H), {{30, 10, 30, 50}, {30, 10, 30, 50}});
    expect(many.size() == 2, "the batch form returns one Stats per segment");
    expectNear(many[0].lambda_min, many[1].lambda_min, 1e-12,
               "and is deterministic");
}

// ---------------------------------------------------------------------------
void testManhattanRecovery() {
    std::printf("estimate: a synthetic Manhattan frame\n");
    lineuq::Camera cam{500.0, 500.0, 320.0, 240.0};

    // A frame rotated away from the camera axes, so recovering it is not
    // trivially the identity.
    const double a = 0.30, b = -0.22, c = 0.15;
    const double ca = std::cos(a), sa = std::sin(a);
    const double cb = std::cos(b), sb = std::sin(b);
    const double cc = std::cos(c), sc = std::sin(c);
    double Rz[3][3] = {{cc, -sc, 0}, {sc, cc, 0}, {0, 0, 1}};
    double Ry[3][3] = {{cb, 0, sb}, {0, 1, 0}, {-sb, 0, cb}};
    double Rx[3][3] = {{1, 0, 0}, {0, ca, -sa}, {0, sa, ca}};
    double T[3][3], R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T[i][j] = 0;
            for (int k = 0; k < 3; ++k) T[i][j] += Rz[i][k] * Ry[k][j];
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R[i][j] = 0;
            for (int k = 0; k < 3; ++k) R[i][j] += T[i][k] * Rx[k][j];
        }

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<lineuq::Segment> segs;
    for (int axis = 0; axis < 3; ++axis) {
        const double d[3] = {R[0][axis], R[1][axis], R[2][axis]};
        for (int k = 0; k < 30; ++k) {
            const double P[3] = {u(rng) * 2.0, u(rng) * 2.0, 6.0 + u(rng) * 2.0};
            const double t = 0.6 + 0.4 * std::fabs(u(rng));
            const double Q[3] = {P[0] + d[0] * t, P[1] + d[1] * t, P[2] + d[2] * t};
            if (P[2] < 1.0 || Q[2] < 1.0) continue;
            segs.push_back({cam.fx * P[0] / P[2] + cam.cx, cam.fy * P[1] / P[2] + cam.cy,
                            cam.fx * Q[0] / Q[2] + cam.cx, cam.fy * Q[1] / Q[2] + cam.cy});
        }
    }
    expect(segs.size() > 60, "the synthetic scene has enough segments");

    const std::vector<lineuq::WeightedLine> lines = lineuq::calibrate(segs, cam);
    const lineuq::Manhattan m = lineuq::estimateManhattan(lines);
    expect(m.ok, "the frame is estimated");

    // Each true axis is matched by some estimated axis, up to sign.
    double worst = 0;
    for (int axis = 0; axis < 3; ++axis) {
        double best = 0;
        for (int k = 0; k < 3; ++k) {
            const double c2 = std::fabs(m.axis[k][0] * R[0][axis] +
                                        m.axis[k][1] * R[1][axis] +
                                        m.axis[k][2] * R[2][axis]);
            if (c2 > best) best = c2;
        }
        const double deg =
            std::acos(best > 1.0 ? 1.0 : best) * 180.0 / 3.14159265358979;
        if (deg > worst) worst = deg;
    }
    expect(worst < 1.0, "every true axis is recovered to within a degree");

    // Determinism: same input, same seed, same answer.
    const lineuq::Manhattan m2 = lineuq::estimateManhattan(lines);
    expectNear(m2.axis[0][0], m.axis[0][0], 0.0, "the estimator is deterministic");

    // axisNearest picks the axis closest to a prior and orients it.
    const double prior[3] = {R[0][1], R[1][1], R[2][1]};
    double v[3];
    expect(lineuq::axisNearest(m, prior, v), "axisNearest succeeds");
    const double c3 = v[0] * prior[0] + v[1] * prior[1] + v[2] * prior[2];
    expect(c3 > 0.999, "and returns the prior's own axis, correctly signed");
}

void testWeightsReachTheEstimator() {
    std::printf("estimate: the weight is the only thing that changes\n");
    lineuq::Camera cam{500.0, 500.0, 320.0, 240.0};
    std::vector<lineuq::Segment> segs;
    for (int k = 0; k < 12; ++k)
        segs.push_back({100.0 + k, 50.0, 100.0 + k, 250.0});

    const std::vector<lineuq::WeightedLine> unit = lineuq::calibrate(segs, cam);
    std::vector<double> w(segs.size(), 0.0);
    for (std::size_t i = 0; i < segs.size(); ++i) w[i] = segs[i].length();
    const std::vector<lineuq::WeightedLine> byLen = lineuq::calibrate(segs, cam, w);

    expect(unit.size() == byLen.size(), "weighting does not change which lines enter");
    expectNear(unit[0].nx, byLen[0].nx, 1e-12, "nor their geometry");
    expect(byLen[0].w > unit[0].w, "only the weight differs");

    // A zero weight drops the line, which is what "this segment is unusable"
    // has to mean to the estimator.
    std::vector<double> z(segs.size(), 1.0);
    z[0] = 0.0;
    expect(lineuq::calibrate(segs, cam, z).size() == segs.size() - 1,
           "a zero-weight segment is dropped");

    // min_length is applied on the segment, not the weight.
    expect(lineuq::calibrate(segs, cam, {}, 1000.0).empty(),
           "min_length filters segments");
}

}  // namespace

int main() {
    std::printf("LineUQ tests\n\n");
    testMeasuresAStraightEdge();
    testDegenerateFitSaturatesUpwards();
    testDirectionFollowsTheEdge();
    testNoiseRaisesLambdaMin();
    testFallbackOnNothing();
    testEmptyAndDegenerateInputs();
    testManhattanRecovery();
    testWeightsReachTheEstimator();

    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
