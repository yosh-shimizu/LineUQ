// LineUQ -- measured line-segment localisation uncertainty.
//
// A line segment fitted by total least squares carries the statistics that say
// how well it was fitted: the number of contributing samples N and the two
// eigenvalues of their covariance.  Most detectors compute something like them
// and then discard them, and the ones they discard are not always a measurement
// of where the underlying intensity ridge is -- a support region's lateral
// spread can record the acceptance rule that drew it instead.
//
// This header is the whole public surface.  Given an image and a pair of
// endpoints, measure() re-measures the ridge under the segment and returns
// (N, lambda_max, lambda_min) together with the derived direction variance and
// weights.  Nothing here knows which detector produced the endpoints.
//
// SPDX-License-Identifier: MIT

#ifndef LINEUQ_LINEUQ_HPP
#define LINEUQ_LINEUQ_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineuq {

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// A non-owning view of an 8-bit greyscale image, row-major, `stride` bytes
/// per row.  Bring your own decoder; LineUQ never allocates your pixels.
struct ImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride = 0;   ///< bytes per row; 0 means `width`

    std::ptrdiff_t rowStride() const { return stride ? stride : width; }
    bool empty() const { return data == nullptr || width <= 0 || height <= 0; }
};

/// A detected segment, in pixel coordinates.  Whole-pixel or sub-pixel
/// endpoints are both fine; they are passed through untouched.
struct Segment {
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    double length() const;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// What one segment's ridge measurement yielded.
///
/// `lambda_max` and `lambda_min` are eigenvalues of the *normalised* covariance
/// of the accepted ridge samples, not of the raw scatter matrix, so both are
/// variances in px^2: a segment of length l whose samples scatter about the
/// fitted line with standard deviation sigma_e has lambda_max ~ l^2/12 and
/// lambda_min ~ sigma_e^2, independently of how many samples were accepted.
/// The convention matters for reading the scalings; it cancels in the weight.
struct Stats {
    double n = 0;              ///< effective number of accepted cross-sections
    double lambda_max = 0;     ///< px^2, along the segment
    double lambda_min = 0;     ///< px^2, across it -- the localisation noise
    double cx = 0, cy = 0;     ///< centroid of the accepted ridge samples
    double dir_x = 1, dir_y = 0;  ///< unit direction of the fitted ridge

    int sections_total = 0;    ///< cross-sections the walk visited
    int sections_used = 0;     ///< of those, admitted into the scatter
    /// True when too few sections survived and the fallback fired: the stats
    /// are then isotropic by construction, which yields the *lowest* weight
    /// rather than an accidentally large one.  Not an error -- it is the pass
    /// reporting that it could not measure this segment, which is a per-segment
    /// statement a support set sized by construction cannot make.
    bool fallback = false;

    /// Plug-in direction-variance proxy, lambda_min / (n * lambda_max).
    /// Exactly zero when the accepted samples were collinear, which is a real
    /// answer and not a failure: nothing about the fit was uncertain.
    double sigma2() const;
    /// Directional information, the reciprocal of sigma2(), and therefore
    /// infinite on a degenerate fit.  Use the weight accessors below to consume
    /// it; they saturate instead.
    double information() const;

    /// Weights saturate here.  A collinear fit carries infinite information,
    /// which no downstream sum can use, so the weight accessors clamp to this
    /// instead -- upwards, because such a segment is the best measured one
    /// present, not the worst.
    static constexpr double kWeightCap = 1e6;

    /// The tempered information weight sqrt(information()), clamped to `cap`.
    /// This is the parameter-free default; see weightPow() before assuming it
    /// is optimal for your data.
    double sqrtFisher(double cap = kWeightCap) const;
    /// The family (N lambda_max / lambda_min)^(q/2), clamped to `cap`: q = 1 is
    /// sqrtFisher(), q = 2 the untempered inverse-variance weight.  Which q is
    /// best is regime-dependent, which is why the statistics are returned and
    /// not only the weight.
    double weightPow(double q, double cap = kWeightCap) const;
};

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Knobs of the ridge measurement.  The defaults are the ones evaluated in the
/// paper; the robustness to them is one-sided, so if you must move the
/// threshold for a different sensor, move it *down*.
struct Params {
    /// Half-width of the search along the segment normal, in pixels.  Widening
    /// it lets neighbouring structure into the cross-sections.
    double search_radius = 2.0;
    /// Admission threshold on the L1 gradient power at the ridge, on the
    /// internal scale of the smoothed image (the Gaussian is scaled by 64).
    double power_threshold = 256.0;
    /// Fewer admitted cross-sections than this and the fallback fires.
    int min_sections = 5;
};

/// Everything the measurement needs from an image, computed once per frame.
/// Building one costs a Gaussian blur and a gradient pass; reuse it across all
/// the segments of that image.
class Frame {
  public:
    Frame() = default;
    explicit Frame(const ImageView& image);

    bool valid() const { return width_ > 0 && height_ > 0; }
    int width() const { return width_; }
    int height() const { return height_; }

    /// Measure one segment against this frame.
    Stats measure(const Segment& s, const Params& p = Params()) const;

  private:
    int width_ = 0, height_ = 0;
    std::vector<float> gx_, gy_, mag_;
};

/// Convenience: build a Frame and measure every segment against it.
std::vector<Stats> measure(const ImageView& image,
                           const std::vector<Segment>& segments,
                           const Params& p = Params());

}  // namespace lineuq

#endif  // LINEUQ_LINEUQ_HPP
