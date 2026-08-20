// Measure the localisation uncertainty of segments some other detector found.
//
//   measure_uncertainty <image> <segments.txt> [--csv out.csv]
//
// segments.txt is one segment per line, "x0 y0 x1 y1", which is what most
// detectors will dump for you in a few lines of code.  Nothing here knows or
// cares which detector that was.
//
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lineuq/lineuq.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "stb_image.h"

int main(int argc, char** argv) {
    const char* image_path = nullptr;
    const char* segs_path = nullptr;
    const char* csv_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else if (!image_path) image_path = argv[i];
        else if (!segs_path) segs_path = argv[i];
    }
    if (!image_path || !segs_path) {
        std::fprintf(stderr,
                     "usage: measure_uncertainty <image> <segments.txt> [--csv out.csv]\n"
                     "       segments.txt: one \"x0 y0 x1 y1\" per line\n");
        return 2;
    }

    int w = 0, h = 0, comp = 0;
    std::uint8_t* pixels = stbi_load(image_path, &w, &h, &comp, 1);
    if (!pixels) {
        std::fprintf(stderr, "cannot read image %s\n", image_path);
        return 1;
    }

    std::vector<lineuq::Segment> segs;
    {
        std::ifstream f(segs_path);
        if (!f) {
            std::fprintf(stderr, "cannot read %s\n", segs_path);
            stbi_image_free(pixels);
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            lineuq::Segment s;
            if (ls >> s.x0 >> s.y0 >> s.x1 >> s.y1) segs.push_back(s);
        }
    }

    lineuq::ImageView view;
    view.data = pixels;
    view.width = w;
    view.height = h;
    view.stride = w;

    // One Frame per image, reused across every segment: the blur and gradient
    // are the per-frame cost, the walk is the per-segment one.
    const lineuq::Frame frame(view);

    std::FILE* out = stdout;
    if (csv_path) {
        out = std::fopen(csv_path, "w");
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", csv_path);
            stbi_image_free(pixels);
            return 1;
        }
    }
    std::fprintf(out, "x0,y0,x1,y1,length,n,lambda_max,lambda_min,sigma2,sqrt_fisher,fallback\n");

    int fallbacks = 0;
    for (const lineuq::Segment& s : segs) {
        const lineuq::Stats st = frame.measure(s);
        fallbacks += st.fallback ? 1 : 0;
        std::fprintf(out, "%.4f,%.4f,%.4f,%.4f,%.4f,%.10g,%.10g,%.10g,%.10g,%.10g,%d\n",
                     s.x0, s.y0, s.x1, s.y1, s.length(), st.n, st.lambda_max,
                     st.lambda_min, st.sigma2(), st.sqrtFisher(), st.fallback ? 1 : 0);
    }
    if (csv_path) std::fclose(out);

    std::fprintf(stderr, "%zu segments, %d could not be measured (%.1f%%)\n", segs.size(),
                 fallbacks, segs.empty() ? 0.0 : 100.0 * fallbacks / double(segs.size()));
    stbi_image_free(pixels);
    return 0;
}
