cmake_minimum_required(VERSION 3.16)
project(lineuq VERSION 0.1.0 LANGUAGES CXX)

option(LINEUQ_BUILD_EXAMPLES "Build the example programs" ON)
option(LINEUQ_BUILD_TESTS "Build the test harness" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- the library
# Depends on nothing but the standard library: it takes a pointer to your
# pixels and gives back numbers.
add_library(lineuq
  src/measure.cpp
  src/estimate.cpp)
add_library(lineuq::lineuq ALIAS lineuq)
target_include_directories(lineuq PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(lineuq PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(lineuq PRIVATE /W4)
else()
  target_compile_options(lineuq PRIVATE -Wall -Wextra -O3)
endif()

# ------------------------------------------------------------------- examples
# Only these read image files, and only they pull in the vendored stb_image.
if(LINEUQ_BUILD_EXAMPLES)
  add_library(lineuq_stb INTERFACE)
  target_include_directories(lineuq_stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

  foreach(ex measure_uncertainty estimate_manhattan)
    add_executable(${ex} examples/${ex}.cpp)
    target_link_libraries(${ex} PRIVATE lineuq lineuq_stb)
  endforeach()
endif()

# ---------------------------------------------------------------------- tests
if(LINEUQ_BUILD_TESTS)
  enable_testing()
  add_executable(lineuq_tests tests/test_lineuq.cpp)
  target_link_libraries(lineuq_tests PRIVATE lineuq)
  add_test(NAME lineuq_tests COMMAND lineuq_tests)
endif()

# -------------------------------------------------------------------- install
include(GNUInstallDirs)
install(TARGETS lineuq EXPORT lineuqTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT lineuqTargets NAMESPACE lineuq::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lineuq)
