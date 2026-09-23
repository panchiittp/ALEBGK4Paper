cmake_minimum_required(VERSION 3.18)

# =============================================================================
# ALE-BGK meshfree kinetic solver - lid-driven cavity cases
#
#   cmake -B build                      serial CPU (OpenMP if available)
#   cmake -B build -DALEBGK_CUDA=ON     GPU backend
#   cmake --build build -j
#
# The GPU backend is the one the 3-D body cases were established on; the CPU
# backend runs the same cases and is the reference for cross-checking.
# =============================================================================
option(ALEBGK_CUDA      "Build the CUDA backend"                OFF)
option(ALEBGK_OPENMP    "OpenMP threading on the host"          ON)
option(ALEBGK_NATIVE    "Optimize for the build machine"        ON)
option(ALEBGK_FAST_MATH "Fast math (UNSAFE for validation)"     OFF)

if(ALEBGK_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Ask the installed GPU what it is, so no -DCMAKE_CUDA_ARCHITECTURES
        # is needed. Falls back to sm_75 with embedded PTX, which JIT-compiles
        # forward onto newer cards.
        execute_process(COMMAND nvidia-smi --query-gpu=compute_cap
                                --format=csv,noheader
                        OUTPUT_VARIABLE ALEBGK_CC
                        RESULT_VARIABLE ALEBGK_CC_RC
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(ALEBGK_CC_RC EQUAL 0 AND ALEBGK_CC MATCHES "^([0-9]+)\\.([0-9]+)")
            set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    endif()
    project(ALEBGK LANGUAGES CXX CUDA)
else()
    project(ALEBGK LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCES
    TestCases/main.cpp
    TestCases/TestCases.cpp
    Initialization/ParticleGeneration.cpp
    Initialization/InitialConditions.cpp
    Initialization/NeighbourSearch.cpp
    Initialization/BodyEmbedding.cpp
    MLSTransport/MLSTransport.cpp
    MomentCalculations/Moments.cpp
    MomentCalculations/Collision.cpp
    MomentCalculations/Conservation.cpp
    BoundaryConditions/WallState.cpp
    BoundaryConditions/DiffuseWall.cpp
    RigidBody/BodyDynamics.cpp
    RigidBody/CloudManagement.cpp
    Solver/Solver.cpp
)
if(ALEBGK_CUDA)
    list(APPEND SOURCES Solver/CudaBackend.cu)
endif()

add_executable(alebgk ${SOURCES})
target_include_directories(alebgk PRIVATE
    Core
    Initialization
    MLSTransport
    MomentCalculations
    BoundaryConditions
    RigidBody
    TestCases
)

if(ALEBGK_CUDA)
    target_compile_definitions(alebgk PRIVATE ALEBGK_WITH_CUDA)
    set_target_properties(alebgk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "Backend: CUDA (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "Backend: serial CPU")
endif()

# OpenMP is applied to C++ translation units only, never to nvcc's host
# compiler line. Gating it on "NOT CUDA" instead - the obvious thing to do -
# silently disables host threading in every GPU build, and the moving-body
# path runs substantially on the host.
if(ALEBGK_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_compile_options(alebgk PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
        if(NOT MSVC)
            target_link_libraries(alebgk PRIVATE OpenMP::OpenMP_CXX)
        endif()
        message(STATUS "OpenMP: enabled (C++ only)")
    endif()
endif()

if(MSVC)
    target_compile_definitions(alebgk PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(alebgk PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<BOOL:${ALEBGK_FAST_MATH}>,/fp:fast,/fp:precise>>)
    if(ALEBGK_NATIVE)
        target_compile_options(alebgk PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
else()
    target_compile_options(alebgk PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>)
    if(ALEBGK_FAST_MATH)
        target_compile_options(alebgk PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
        message(WARNING "ALEBGK_FAST_MATH reorders reductions and drops "
                        "NaN handling; do not use for validation runs")
    endif()
    if(ALEBGK_NATIVE AND NOT ALEBGK_CUDA)
        target_compile_options(alebgk PRIVATE -march=native)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(alebgk PRIVATE stdc++fs)
    endif()
endif()

if(ALEBGK_CUDA)
    target_compile_options(alebgk PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>)
endif()

set_target_properties(alebgk PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

message(STATUS "")
message(STATUS "=== ALE-BGK cavity solver ===")
message(STATUS "Build: cmake -B build -DALEBGK_CUDA=ON && cmake --build build -j")
message(STATUS "Run:   build/bin/alebgk <case>   (no arguments lists the cases)")
message(STATUS "")
