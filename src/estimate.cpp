// Weighted Manhattan-frame estimation.
//
// The structure is deliberately ordinary: candidate directions from line pairs,
// ranked by inlier support, then a weighted total-least-squares refit of each
// axis against the normals inside its band, iterated.  The only place the
// per-segment weight enters is refitAxis(), which is what makes "unit vs length
// vs measured uncertainty" a controlled comparison -- nothing else about the
// estimator changes with it.
//
// SPDX-License-Identifier: MIT

#include "lineuq/estimate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace lineuq {
namespace {

struct V3 {
    double x = 0, y = 0, z = 0;
};

double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(const V3& a) { return std::sqrt(dot(a, a)); }
V3 normalize(const V3& a) {
    const double n = norm(a);
    return n > 1e-12 ? V3{a.x / n, a.y / n, a.z / n} : V3{0, 0, 0};
}

/// Jacobi eigen-decomposition of a symmetric 3x3.  Eigenvectors are the columns
/// of V.  Three-by-three symmetric is small enough that a closed form exists,
/// but Jacobi is the one that stays accurate on the near-degenerate scatters a
/// short line bundle produces.
void jacobiEigen(const double A[3][3], double w[3], double V[3][3]) {
    double a[3][3];
    std::memcpy(a, A, sizeof(a));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; ++sweep) {
        const double off =
            std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
        if (off < 1e-14) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-18) continue;
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1.0 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 3; ++i) w[i] = a[i][i];
}

V3 smallestEigenvector(const double S[3][3]) {
    double w[3], V[3][3];
    jacobiEigen(S, w, V);
    int m = 0;
    if (w[1] < w[m]) m = 1;
    if (w[2] < w[m]) m = 2;
    return normalize({V[0][m], V[1][m], V[2][m]});
}

V3 nOf(const WeightedLine& l) { return {l.nx, l.ny, l.nz}; }

V3 refit(const std::vector<WeightedLine>& lines, const V3& d, double tau) {
    double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    int cnt = 0;
    for (const WeightedLine& l : lines) {
        const V3 n = nOf(l);
        if (std::fabs(dot(n, d)) > tau) continue;
        const double w = l.w;
        S[0][0] += w * n.x * n.x; S[0][1] += w * n.x * n.y; S[0][2] += w * n.x * n.z;
        S[1][1] += w * n.y * n.y; S[1][2] += w * n.y * n.z; S[2][2] += w * n.z * n.z;
        ++cnt;
    }
    if (cnt < 2) return d;
    S[1][0] = S[0][1]; S[2][0] = S[0][2]; S[2][1] = S[1][2];
    return smallestEigenvector(S);
}

double axisScore(const std::vector<WeightedLine>& lines, const V3& d, double tau) {
    double s = 0;
    for (const WeightedLine& l : lines)
        if (std::fabs(dot(nOf(l), d)) < tau) s += l.w;
    return s;
}

double frameScore(const std::vector<WeightedLine>& lines, const V3 D[3], double tau) {
    double s = 0;
    for (const WeightedLine& l : lines) {
        const V3 n = nOf(l);
        double best = 1e9;
        for (int k = 0; k < 3; ++k) best = std::min(best, std::fabs(dot(n, D[k])));
        if (best < tau) s += l.w;
    }
    return s;
}

/// Gram-Schmidt plus a cross product: makes a triad orthonormal while keeping
/// the first axis exactly and the second as close as it can.
void orthonormalize(V3 a, V3 b, V3 out[3]) {
    a = normalize(a);
    b = normalize({b.x - dot(b, a) * a.x, b.y - dot(b, a) * a.y, b.z - dot(b, a) * a.z});
    out[0] = a;
    out[1] = b;
    out[2] = normalize(cross(a, b));
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<WeightedLine> calibrate(const std::vector<Segment>& segments,
                                    const Camera& camera,
                                    const std::vector<double>& weights,
                                    double min_length) {
    const bool have_w = !weights.empty();
    std::vector<WeightedLine> out;
    out.reserve(segments.size());
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const Segment& s = segments[i];
        if (s.length() < min_length) continue;
        auto ray = [&](double x, double y) {
            return V3{(x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, 1.0};
        };
        const V3 n = cross(ray(s.x0, s.y0), ray(s.x1, s.y1));
        if (norm(n) < 1e-9) continue;
        const V3 u = normalize(n);
        const double w = have_w ? (i < weights.size() ? weights[i] : 0.0) : 1.0;
        if (!(w > 0)) continue;
        out.push_back({u.x, u.y, u.z, w});
    }
    return out;
}

void refitAxis(const std::vector<WeightedLine>& lines, double tau,
               const double d_in[3], double d_out[3]) {
    const V3 r = refit(lines, normalize({d_in[0], d_in[1], d_in[2]}), tau);
    d_out[0] = r.x; d_out[1] = r.y; d_out[2] = r.z;
}

Manhattan estimateManhattan(const std::vector<WeightedLine>& lines,
                            const EstimatorParams& p) {
    Manhattan out;
    const int N = int(lines.size());
    if (N < 8) return out;

    // Candidate directions: a vanishing direction shared by two segments is
    // perpendicular to both their interpretation-plane normals.
    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> pick(0, N - 1);
    std::vector<V3> cand;
    const int pairs = std::min(p.max_pairs, N * (N - 1) / 2);
    cand.reserve(std::size_t(pairs));
    for (int t = 0; t < pairs; ++t) {
        const int i = pick(rng), j = pick(rng);
        if (i == j) continue;
        const V3 d = cross(nOf(lines[i]), nOf(lines[j]));
        if (norm(d) < 1e-6) continue;
        cand.push_back(normalize(d));
    }
    if (cand.empty()) return out;

    std::stable_sort(cand.begin(), cand.end(), [&](const V3& a, const V3& b) {
        return axisScore(lines, a, p.tau) > axisScore(lines, b, p.tau);
    });

    // Multi-start: the strongest few *distinct* candidates, so a scene whose
    // best-supported direction is not part of the true frame still has a way in.
    std::vector<V3> seeds;
    for (const V3& d : cand) {
        bool dup = false;
        for (const V3& s : seeds)
            if (std::fabs(dot(d, s)) > 0.99) { dup = true; break; }
        if (!dup) seeds.push_back(d);
        if (int(seeds.size()) >= p.max_seeds) break;
    }

    double best = -1;
    for (const V3& d1 : seeds) {
        // Best-supported candidate that is roughly perpendicular to the seed.
        V3 d2{0, 0, 0};
        double best2 = -1;
        for (const V3& d : cand) {
            if (std::fabs(dot(d, d1)) > 0.26) continue;   // within ~15 deg of perpendicular
            const double s = axisScore(lines, d, p.tau);
            if (s > best2) { best2 = s; d2 = d; }
        }
        if (best2 < 0) {
            const V3 t = std::fabs(d1.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0};
            d2 = normalize(cross(d1, t));
        }

        V3 tri[3];
        orthonormalize(d1, d2, tri);
        for (int iter = 0; iter < p.refine_iters; ++iter) {
            V3 R[3] = {refit(lines, tri[0], p.tau), refit(lines, tri[1], p.tau),
                       refit(lines, tri[2], p.tau)};
            double sc[3];
            for (int a = 0; a < 3; ++a) sc[a] = axisScore(lines, R[a], p.tau);
            int order[3] = {0, 1, 2};
            std::stable_sort(order, order + 3,
                             [&](int a, int b) { return sc[a] > sc[b]; });
            orthonormalize(R[order[0]], R[order[1]], tri);
        }

        const double sc = frameScore(lines, tri, p.tau);
        if (sc > best) {
            best = sc;
            for (int k = 0; k < 3; ++k) {
                out.axis[k][0] = tri[k].x;
                out.axis[k][1] = tri[k].y;
                out.axis[k][2] = tri[k].z;
            }
            out.ok = true;
            out.support = sc;
        }
    }
    return out;
}

bool axisNearest(const Manhattan& frame, const double prior[3], double out[3]) {
    if (!frame.ok) return false;
    const V3 pr = normalize({prior[0], prior[1], prior[2]});
    if (norm(pr) < 0.5) return false;
    int best = 0;
    double best_abs = -1;
    for (int k = 0; k < 3; ++k) {
        const double c = std::fabs(frame.axis[k][0] * pr.x + frame.axis[k][1] * pr.y +
                                   frame.axis[k][2] * pr.z);
        if (c > best_abs) { best_abs = c; best = k; }
    }
    const double sign =
        (frame.axis[best][0] * pr.x + frame.axis[best][1] * pr.y +
         frame.axis[best][2] * pr.z) < 0 ? -1.0 : 1.0;
    for (int i = 0; i < 3; ++i) out[i] = sign * frame.axis[best][i];
    return true;
}

}  // namespace lineuq
