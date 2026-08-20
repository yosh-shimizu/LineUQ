// Ridge re-measurement: the posterior pass.
//
// What the statistics describe is the intensity ridge lying under a pair of
// endpoints, not the procedure that found them, so they can be re-estimated
// from image evidence local to the reported segment.  We smooth and
// differentiate once per frame, then walk the segment's dominant axis one pixel
// at a time, search along the normal for the gradient ridge, refine each hit to
// sub-pixel with a parabola, and accumulate the scatter of the accepted
// cross-sections.  Sections whose peak sits at the edge of the search window
// are mis-localised and are dropped as gaps.
//
// SPDX-License-Identifier: MIT

#include "lineuq/lineuq.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace lineuq {
namespace {

// ---------------------------------------------------------------------------
// Fixed-point Gaussian, {16,64,96,64,16} separable with one >>10 rescale.
//
// The result is the smoothed image scaled by 64, which is the scale
// Params::power_threshold is expressed on.  Rows and columns within 2 px of the
// border are zero: the kernel has no support there and inventing one would put
// a fabricated gradient under any segment that reaches the frame edge.
// ---------------------------------------------------------------------------
constexpr int kW0 = 16, kW1 = 64, kW2 = 96, kShift = 10;

void gaussianBlur(const ImageView& src, std::vector<std::uint16_t>& out) {
    const int w = src.width, h = src.height;
    const std::ptrdiff_t stride = src.rowStride();
    out.assign(std::size_t(w) * h, 0);

    const std::vector<std::uint8_t> zero(std::size_t(w), 0);
    auto row = [&](int y) -> const std::uint8_t* {
        return (y >= 0 && y < h) ? src.data + std::ptrdiff_t(y) * stride : zero.data();
    };

    std::vector<std::uint32_t> vert(static_cast<std::size_t>(w), 0u);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* r0 = row(y - 2);
        const std::uint8_t* r1 = row(y - 1);
        const std::uint8_t* r2 = row(y);
        const std::uint8_t* r3 = row(y + 1);
        const std::uint8_t* r4 = row(y + 2);
        for (int x = 0; x < w; ++x)                       // <= 255 * 256 = 65280
            vert[std::size_t(x)] = std::uint32_t(kW0 * (r0[x] + r4[x]) +
                                                 kW1 * (r1[x] + r3[x]) + kW2 * r2[x]);
        if (y < 2 || y >= h - 2 || w < 5) continue;       // left zeroed by assign()
        std::uint16_t* o = &out[std::size_t(y) * w];
        for (int x = 2; x < w - 2; ++x) {
            const std::uint32_t s = kW0 * (vert[x - 2] + vert[x + 2]) +
                                    kW1 * (vert[x - 1] + vert[x + 1]) + kW2 * vert[x];
            o[x] = std::uint16_t(s >> kShift);            // <= 16320 = 255 * 64
        }
    }
}

// 2x2 gradient on the smoothed image, sign kept.  Reading one past the right or
// bottom edge yields zero, which matches the border the blur already imposes.
void signedGradient(const std::vector<std::uint16_t>& g, int w, int h,
                    std::vector<float>& gx, std::vector<float>& gy,
                    std::vector<float>& mag) {
    const std::size_t n = std::size_t(w) * h;
    gx.assign(n, 0.f);
    gy.assign(n, 0.f);
    mag.assign(n, 0.f);
    for (int y = 0; y < h; ++y) {
        const std::uint16_t* g0 = &g[std::size_t(y) * w];
        const std::uint16_t* g1 = (y + 1 < h) ? &g[std::size_t(y + 1) * w] : nullptr;
        for (int x = 0; x < w; ++x) {
            const int g00 = g0[x];
            const int g10 = (x + 1 < w) ? g0[x + 1] : 0;
            const int g01 = g1 ? g1[x] : 0;
            const int g11 = (g1 && x + 1 < w) ? g1[x + 1] : 0;
            const float dx = float((g10 + g11) - (g00 + g01));
            const float dy = float((g01 + g11) - (g00 + g10));
            const std::size_t i = std::size_t(y) * w + x;
            gx[i] = dx;
            gy[i] = dy;
            mag[i] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

float sampleBilinear(const std::vector<float>& m, int w, int h, float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > float(w) - 1.001f) x = float(w) - 1.001f;
    if (y > float(h) - 1.001f) y = float(h) - 1.001f;
    const int x0 = int(x), y0 = int(y);
    const float fx = x - float(x0), fy = y - float(y0);
    const float* r0 = &m[std::size_t(y0) * w];
    const float* r1 = &m[std::size_t(y0 + 1) * w];
    return (r0[x0] * (1 - fx) + r0[x0 + 1] * fx) * (1 - fy) +
           (r1[x0] * (1 - fx) + r1[x0 + 1] * fx) * fy;
}

}  // namespace

// ---------------------------------------------------------------------------
// Segment / Stats
// ---------------------------------------------------------------------------
double Segment::length() const { return std::hypot(x1 - x0, y1 - y0); }

double Stats::sigma2() const {
    if (n <= 0 || lambda_max <= 0 || lambda_min < 0) return 0.0;
    return lambda_min / (n * lambda_max);
}

double Stats::information() const {
    if (n <= 0 || lambda_max <= 0) return 0.0;   // nothing was measured at all
    const double s2 = sigma2();
    return s2 > 0 ? 1.0 / s2 : std::numeric_limits<double>::infinity();
}

double Stats::sqrtFisher(double cap) const { return weightPow(1.0, cap); }

double Stats::weightPow(double q, double cap) const {
    // The degenerate case is deliberately the *top* of the range: samples that
    // were exactly collinear are the best-localised evidence in the frame.
    if (n <= 0 || lambda_max <= 0 || lambda_min <= 1e-12) return cap;
    const double w = std::pow(n * lambda_max / lambda_min, 0.5 * q);
    return w > cap ? cap : w;
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------
Frame::Frame(const ImageView& image) {
    if (image.empty()) return;
    width_ = image.width;
    height_ = image.height;
    std::vector<std::uint16_t> smooth;
    gaussianBlur(image, smooth);
    signedGradient(smooth, width_, height_, gx_, gy_, mag_);
}

Stats Frame::measure(const Segment& s, const Params& p) const {
    Stats out;
    const double dx = s.x1 - s.x0, dy = s.y1 - s.y0;
    const double L = std::hypot(dx, dy);
    const int steps = int(std::lround(std::fmax(std::fabs(dx), std::fabs(dy)))) + 1;

    double count = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    if (valid() && L > 1e-6 && steps >= 2) {
        const double ux = dx / L, uy = dy / L;   // along the segment
        const double nx = -uy, ny = ux;          // across it
        const int nlat = int(std::lround(p.search_radius * 2));  // 0.5 px grid

        for (int i = 0; i < steps; ++i) {
            ++out.sections_total;
            const double t = L * i / (steps - 1);
            const double px = s.x0 + ux * t, py = s.y0 + uy * t;

            int best = 0;
            float best_m = -1.f;
            for (int k = -nlat; k <= nlat; ++k) {
                const float m = sampleBilinear(mag_, width_, height_,
                                               float(px + nx * 0.5 * k),
                                               float(py + ny * 0.5 * k));
                if (m > best_m) { best_m = m; best = k; }
            }
            // The ridge sitting on the window edge means the peak is somewhere
            // outside it: the section is mis-localised, so it is a gap and not
            // a measurement.
            if (best == -nlat || best == nlat) continue;

            const float qx0 = float(px + nx * 0.5 * best);
            const float qy0 = float(py + ny * 0.5 * best);
            const float agx = sampleBilinear(gx_, width_, height_, qx0, qy0);
            const float agy = sampleBilinear(gy_, width_, height_, qx0, qy0);
            const double power = (std::fabs(agx) + std::fabs(agy) + 1.0) * 0.5;
            if (power < p.power_threshold) continue;

            const float mm = sampleBilinear(mag_, width_, height_,
                                            float(px + nx * 0.5 * (best - 1)),
                                            float(py + ny * 0.5 * (best - 1)));
            const float mp = sampleBilinear(mag_, width_, height_,
                                            float(px + nx * 0.5 * (best + 1)),
                                            float(py + ny * 0.5 * (best + 1)));
            double off = 0.5 * best;
            const double den = 2.0 * best_m - mm - mp;
            if (den > 1e-9) {
                double o = 0.5 * (mp - mm) / den;        // in 0.5 px grid units
                if (o > 0.5) o = 0.5;
                if (o < -0.5) o = -0.5;
                off += 0.5 * o;
            }
            const double qx = px + nx * off, qy = py + ny * off;

            count += 1;
            sx += qx; sy += qy;
            sxx += qx * qx; sxy += qx * qy; syy += qy * qy;
            ++out.sections_used;
        }
    }

    if (count >= p.min_sections) {
        out.n = count;
        out.cx = sx / count;
        out.cy = sy / count;
        const double cxx = sxx / count - out.cx * out.cx;
        const double cxy = sxy / count - out.cx * out.cy;
        const double cyy = syy / count - out.cy * out.cy;
        const double tr = cxx + cyy;
        const double rt = std::sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
        out.lambda_max = (tr + rt) / 2;
        out.lambda_min = (tr - rt) / 2;
        if (out.lambda_min < 0) out.lambda_min = 0;
        const double th = 0.5 * std::atan2(2 * cxy, cxx - cyy);
        out.dir_x = std::cos(th);
        out.dir_y = std::sin(th);
    } else {
        // Not enough gradient support to measure this segment.  Reporting an
        // isotropic scatter gives it the lowest weight in the family, which is
        // the safe direction: a segment we could not measure should not be
        // trusted more than one we could.
        out.fallback = true;
        out.n = count >= 1 ? count : 1;
        out.cx = (s.x0 + s.x1) / 2;
        out.cy = (s.y0 + s.y1) / 2;
        out.dir_x = L > 1e-6 ? dx / L : 1.0;
        out.dir_y = L > 1e-6 ? dy / L : 0.0;
        out.lambda_max = out.lambda_min = std::fmax(L * L / 12.0, 1e-3);
    }
    return out;
}

// ---------------------------------------------------------------------------
std::vector<Stats> measure(const ImageView& image,
                           const std::vector<Segment>& segments,
                           const Params& p) {
    const Frame frame(image);
    std::vector<Stats> out;
    out.reserve(segments.size());
    for (const Segment& s : segments) out.push_back(frame.measure(s, p));
    return out;
}

}  // namespace lineuq
