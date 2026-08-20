// Estimate a Manhattan frame three ways from the same segments, so the effect
// of the weight is the only thing that varies.
//
//   estimate_manhattan <image> <segments.txt> <fx> <fy> <cx> <cy>
//
// Prints the three estimated axes under unit votes, length weighting, and the
// measured uncertainty weight.  If the three answers differ, that difference is
// the entire subject of the paper this library comes from.
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lineuq/estimate.hpp"
#include "lineuq/lineuq.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "stb_image.h"

namespace {

void report(const char* name, const std::vector<lineuq::Segment>& segs,
            const lineuq::Camera& cam, const std::vector<double>& w) {
    const std::vector<lineuq::WeightedLine> lines = lineuq::calibrate(segs, cam, w);
    const lineuq::Manhattan m = lineuq::estimateManhattan(lines);
    if (!m.ok) {
        std::printf("%-22s (no solution from %zu lines)\n", name, lines.size());
        return;
    }
    std::printf("%-22s support %8.1f   axes", name, m.support);
    for (int k = 0; k < 3; ++k)
        std::printf("  [%+.4f %+.4f %+.4f]", m.axis[k][0], m.axis[k][1], m.axis[k][2]);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: estimate_manhattan <image> <segments.txt> fx fy cx cy\n");
        return 2;
    }
    const char* image_path = argv[1];
    const char* segs_path = argv[2];
    lineuq::Camera cam{std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]),
                       std::atof(argv[6])};

    int w = 0, h = 0, comp = 0;
    std::uint8_t* pixels = stbi_load(image_path, &w, &h, &comp, 1);
    if (!pixels) {
        std::fprintf(stderr, "cannot read image %s\n", image_path);
        return 1;
    }

    std::vector<lineuq::Segment> segs;
    {
        std::ifstream f(segs_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            lineuq::Segment s;
            if (ls >> s.x0 >> s.y0 >> s.x1 >> s.y1) segs.push_back(s);
        }
    }
    if (segs.empty()) {
        std::fprintf(stderr, "no segments in %s\n", segs_path);
        stbi_image_free(pixels);
        return 1;
    }

    lineuq::ImageView view;
    view.data = pixels;
    view.width = w;
    view.height = h;
    view.stride = w;

    const lineuq::Frame frame(view);
    std::vector<double> w_len(segs.size()), w_unc(segs.size());
    int fallbacks = 0;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        const lineuq::Stats st = frame.measure(segs[i]);
        fallbacks += st.fallback ? 1 : 0;
        w_len[i] = segs[i].length();
        w_unc[i] = st.sqrtFisher();
    }
    std::printf("%zu segments, %d unmeasurable\n\n", segs.size(), fallbacks);

    report("unit votes", segs, cam, {});
    report("length", segs, cam, w_len);
    report("measured uncertainty", segs, cam, w_unc);

    stbi_image_free(pixels);
    return 0;
}
