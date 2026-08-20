# The extension is declared here rather than in pyproject.toml because it has to
# name the C++ sources, which are shared with the CMake build -- there is one
# implementation of the measurement, not a C++ one and a Python one.
#
# SPDX-License-Identifier: MIT

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "lineuq._core",
    sources=[
        "python/src/bindings.cpp",
        "src/measure.cpp",
        "src/estimate.cpp",
    ],
    include_dirs=["include"],
    cxx_std=17,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
