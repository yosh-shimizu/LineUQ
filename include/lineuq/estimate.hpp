// Uncertainty-aware geometric estimation from line segments.
//
// The point of measuring a per-segment localisation uncertainty is that some
// downstream estimator consumes it, so LineUQ ships the consuming side too: a
// weighted Manhattan-frame / vanishing-point fit in which the only thing that
// changes between "unit votes", "length" and "measured uncertainty" is the
// weight vector you pass in.
//
// SPDX-License-Identifier: MIT

#ifndef LINEUQ_ESTIMATE_HPP
#define LINEUQ_ESTIMATE_HPP

#include <vector>

#include "lineuq/lineuq.hpp"

namespace lineuq {

/// Pinhole intrinsics.  Only fx, fy and the principal point are used; the
/// estimator is calibrated, so an image with unknown intrinsics needs a
/// different method.
struct Camera {
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

/// A segment reduced to what the estimator actually consumes: the unit normal
/// of its interpretation plane (any vanishing direction d shared by the segment
/// satisfies d . n = 0), and one scalar weight.
struct WeightedLine {
    double nx = 0, ny = 0, nz = 0;
    double w = 1.0;
};

/// Project segments into interpretation-plane normals.
///
/// `weights` is either empty (every segment counts once), or one weight per
/// segment.  Pass `Stats::sqrtFisher()` for the measured uncertainty weight,
/// `Segment::length()` for the length baseline, or anything else you want to
/// test -- the estimator does not care where the number came from.
std::vector<WeightedLine> calibrate(const std::vector<Segment>& segments,
                                    const Camera& camera,
                                    const std::vector<double>& weights = {},
                                    double min_length = 0.0);

struct EstimatorParams {
    /// Inlier half-band on |n . d|, in radians of sine.  A segment counts
    /// towards an axis when its normal is within this of perpendicular.
    double tau = 0.035;      ///< sin(2 degrees)
    int max_pairs = 4000;    ///< candidate directions drawn from line pairs
    int max_seeds = 10;      ///< distinct first-axis seeds kept
    int refine_iters = 8;    ///< weighted refits per seed
    unsigned seed = 12345;   ///< the candidate draw is seeded, so runs repeat
};

struct Manhattan {
    bool ok = false;
    /// The three mutually orthogonal directions, as unit vectors in camera
    /// coordinates.  These are the vanishing directions: the image-plane
    /// vanishing point of axis k is the projection of axis[k].
    double axis[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double support = 0;      ///< total weight explained by the triad
};

/// Fit an orthogonal Manhattan frame. Deterministic for a given input and seed.
Manhattan estimateManhattan(const std::vector<WeightedLine>& lines,
                            const EstimatorParams& p = EstimatorParams());

/// Re-fit one axis as the direction most perpendicular to the weighted scatter
/// of the normals inside its band.  This is the one line where the weight acts,
/// and it is exposed so that a caller with its own axis hypothesis can use the
/// weighting without the search.
void refitAxis(const std::vector<WeightedLine>& lines, double tau,
               const double d_in[3], double d_out[3]);

/// The axis of `frame` closest to `prior` (a rough vertical, say), with its
/// sign chosen to agree with the prior.
///
/// Note this is a convenience on top of the triad, not the streaming
/// zenith/Manhattan hybrid the paper measures gravity with; that estimator has
/// a vertical-only fast path this library does not reproduce.
bool axisNearest(const Manhattan& frame, const double prior[3], double out[3]);

}  // namespace lineuq

#endif  // LINEUQ_ESTIMATE_HPP
