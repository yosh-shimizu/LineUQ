// Python bindings for LineUQ.
//
// The binding is array-oriented rather than object-oriented: measuring one
// segment at a time across the Python boundary would spend more time in the
// boundary than in the measurement.  Everything takes and returns numpy arrays,
// and the ergonomics live in the Python layer on top.
//
// SPDX-License-Identifier: MIT

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "lineuq/estimate.hpp"
#include "lineuq/lineuq.hpp"

namespace py = pybind11;

namespace {

using ArrayU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

lineuq::ImageView viewOf(const ArrayU8& image) {
    const auto buf = image.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("image must be a 2-D array of uint8 (H, W)");
    lineuq::ImageView v;
    v.data = static_cast<const std::uint8_t*>(buf.ptr);
    v.height = int(buf.shape[0]);
    v.width = int(buf.shape[1]);
    v.stride = v.width;   // forcecast + c_style guarantees this
    return v;
}

std::vector<lineuq::Segment> segmentsOf(const ArrayD& segments) {
    const auto buf = segments.request();
    if (buf.ndim != 2 || buf.shape[1] != 4)
        throw std::invalid_argument("segments must have shape (N, 4): x0 y0 x1 y1");
    const double* p = static_cast<const double*>(buf.ptr);
    std::vector<lineuq::Segment> out(std::size_t(buf.shape[0]));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = {p[4 * i + 0], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]};
    return out;
}

/// Statistics for a whole segment set, as one dict of numpy columns.  The
/// Python layer wraps this in a class; keeping the C++ side a dict avoids
/// binding a struct-of-arrays type twice.
py::dict statsToDict(const std::vector<lineuq::Stats>& stats) {
    const std::size_t n = stats.size();
    ArrayD n_arr(n), lmax(n), lmin(n), cx(n), cy(n), dx(n), dy(n);
    py::array_t<std::int64_t> total(n), used(n);
    py::array_t<bool> fb(n);

    auto N = n_arr.mutable_unchecked<1>();
    auto A = lmax.mutable_unchecked<1>();
    auto B = lmin.mutable_unchecked<1>();
    auto CX = cx.mutable_unchecked<1>();
    auto CY = cy.mutable_unchecked<1>();
    auto DX = dx.mutable_unchecked<1>();
    auto DY = dy.mutable_unchecked<1>();
    auto T = total.mutable_unchecked<1>();
    auto U = used.mutable_unchecked<1>();
    auto F = fb.mutable_unchecked<1>();

    for (std::size_t i = 0; i < n; ++i) {
        const lineuq::Stats& s = stats[i];
        N(i) = s.n;
        A(i) = s.lambda_max;
        B(i) = s.lambda_min;
        CX(i) = s.cx;
        CY(i) = s.cy;
        DX(i) = s.dir_x;
        DY(i) = s.dir_y;
        T(i) = s.sections_total;
        U(i) = s.sections_used;
        F(i) = s.fallback;
    }

    py::dict d;
    d["n"] = n_arr;
    d["lambda_max"] = lmax;
    d["lambda_min"] = lmin;
    d["cx"] = cx;
    d["cy"] = cy;
    d["dir_x"] = dx;
    d["dir_y"] = dy;
    d["sections_total"] = total;
    d["sections_used"] = used;
    d["fallback"] = fb;
    return d;
}

/// The weight family, evaluated straight from the three statistics.
///
/// This is the entry point a detector that already accumulates its own moments
/// should use: it needs no image and no Frame, so a native supplier and the
/// posterior pass produce interchangeable weights.
ArrayD weightPow(const ArrayD& n, const ArrayD& lambda_max, const ArrayD& lambda_min,
                 double q, double cap) {
    const auto bn = n.request(), ba = lambda_max.request(), bb = lambda_min.request();
    if (bn.size != ba.size || bn.size != bb.size)
        throw std::invalid_argument("n, lambda_max and lambda_min must be the same length");
    const double* pn = static_cast<const double*>(bn.ptr);
    const double* pa = static_cast<const double*>(ba.ptr);
    const double* pb = static_cast<const double*>(bb.ptr);

    ArrayD out(bn.size);
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < bn.size; ++i) {
        lineuq::Stats s;
        s.n = pn[i];
        s.lambda_max = pa[i];
        s.lambda_min = pb[i];
        o(i) = s.weightPow(q, cap);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "LineUQ core: measured line-segment localisation uncertainty.";

    py::class_<lineuq::Params>(m, "Params")
        .def(py::init<>())
        .def(py::init([](double r, double t, int k) {
                 lineuq::Params p;
                 p.search_radius = r;
                 p.power_threshold = t;
                 p.min_sections = k;
                 return p;
             }),
             py::arg("search_radius") = 2.0, py::arg("power_threshold") = 256.0,
             py::arg("min_sections") = 5)
        .def_readwrite("search_radius", &lineuq::Params::search_radius)
        .def_readwrite("power_threshold", &lineuq::Params::power_threshold)
        .def_readwrite("min_sections", &lineuq::Params::min_sections);

    py::class_<lineuq::Frame>(m, "Frame")
        .def(py::init([](const ArrayU8& image) {
                 // The Frame copies what it needs out of the image, so the
                 // caller's array is free to go away afterwards.
                 return new lineuq::Frame(viewOf(image));
             }),
             py::arg("image"))
        .def_property_readonly("width", &lineuq::Frame::width)
        .def_property_readonly("height", &lineuq::Frame::height)
        .def_property_readonly("valid", &lineuq::Frame::valid)
        .def(
            "measure",
            [](const lineuq::Frame& f, const ArrayD& segments, const lineuq::Params& p) {
                const std::vector<lineuq::Segment> segs = segmentsOf(segments);
                std::vector<lineuq::Stats> out;
                out.reserve(segs.size());
                {
                    py::gil_scoped_release nogil;
                    for (const lineuq::Segment& s : segs) out.push_back(f.measure(s, p));
                }
                return statsToDict(out);
            },
            py::arg("segments"), py::arg("params") = lineuq::Params());

    m.def(
        "measure",
        [](const ArrayU8& image, const ArrayD& segments, const lineuq::Params& p) {
            const lineuq::Frame frame(viewOf(image));
            const std::vector<lineuq::Segment> segs = segmentsOf(segments);
            std::vector<lineuq::Stats> out;
            out.reserve(segs.size());
            {
                py::gil_scoped_release nogil;
                for (const lineuq::Segment& s : segs) out.push_back(frame.measure(s, p));
            }
            return statsToDict(out);
        },
        py::arg("image"), py::arg("segments"), py::arg("params") = lineuq::Params());

    m.def("weight_pow", &weightPow, py::arg("n"), py::arg("lambda_max"),
          py::arg("lambda_min"), py::arg("q") = 1.0,
          py::arg("cap") = lineuq::Stats::kWeightCap);

    m.attr("WEIGHT_CAP") = lineuq::Stats::kWeightCap;

    // ---------------------------------------------------------------- estimator
    py::class_<lineuq::EstimatorParams>(m, "EstimatorParams")
        .def(py::init<>())
        .def_readwrite("tau", &lineuq::EstimatorParams::tau)
        .def_readwrite("max_pairs", &lineuq::EstimatorParams::max_pairs)
        .def_readwrite("max_seeds", &lineuq::EstimatorParams::max_seeds)
        .def_readwrite("refine_iters", &lineuq::EstimatorParams::refine_iters)
        .def_readwrite("seed", &lineuq::EstimatorParams::seed);

    m.def(
        "estimate_manhattan",
        [](const ArrayD& segments, std::array<double, 4> camera,
           py::object weights, double min_length, const lineuq::EstimatorParams& ep) {
            lineuq::Camera cam{camera[0], camera[1], camera[2], camera[3]};
            const std::vector<lineuq::Segment> segs = segmentsOf(segments);

            std::vector<double> w;
            if (!weights.is_none()) {
                const ArrayD a = weights.cast<ArrayD>();
                const auto b = a.request();
                if (b.size != py::ssize_t(segs.size()))
                    throw std::invalid_argument("weights must have one entry per segment");
                const double* p = static_cast<const double*>(b.ptr);
                w.assign(p, p + b.size);
            }

            const std::vector<lineuq::WeightedLine> lines =
                lineuq::calibrate(segs, cam, w, min_length);
            lineuq::Manhattan man;
            {
                py::gil_scoped_release nogil;
                man = lineuq::estimateManhattan(lines, ep);
            }

            ArrayD axes({py::ssize_t(3), py::ssize_t(3)});
            auto A = axes.mutable_unchecked<2>();
            for (py::ssize_t k = 0; k < 3; ++k)
                for (py::ssize_t i = 0; i < 3; ++i) A(k, i) = man.axis[k][i];

            py::dict d;
            d["ok"] = man.ok;
            d["axes"] = axes;
            d["support"] = man.support;
            d["lines_used"] = py::int_(py::ssize_t(lines.size()));
            return d;
        },
        py::arg("segments"), py::arg("camera"), py::arg("weights") = py::none(),
        py::arg("min_length") = 0.0,
        py::arg("params") = lineuq::EstimatorParams());
}
