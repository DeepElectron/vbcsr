cmake_minimum_required(VERSION 3.15)
project(vbcsr LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies

# Find dependencies
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# BLAS Integer Size Handling
option(VBCSR_USE_ILP64 "Use 64-bit integers for BLAS/LAPACK interfaces" OFF)

# Find BLAS and LAPACK
# We prefer MKL if available, but via standard FindBLAS
if(NOT BLAS_LIBRARIES)
    find_package(BLAS REQUIRED)
endif()

if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    
    # Auto-detect ILP64 if not explicitly set
    if(NOT DEFINED CACHE{VBCSR_USE_ILP64})
        string(TOLOWER "${BLAS_LIBRARIES}" BLAS_LIBRARIES_LOWER)
        if(BLAS_LIBRARIES_LOWER MATCHES "ilp64")
            set(VBCSR_USE_ILP64 ON CACHE BOOL "Use 64-bit integers for BLAS/LAPACK interfaces" FORCE)
            message(STATUS "Auto-detected ILP64 BLAS, enabling VBCSR_USE_ILP64")
        endif()
    endif()

    # Check for MKL
    string(TOLOWER "${BLAS_LIBRARIES}" BLAS_LIBRARIES_LOWER)
    if(BLAS_LIBRARIES_LOWER MATCHES "mkl" OR BLAS_LIBRARIES_LOWER MATCHES "intel")
        set(VBCSR_HAS_MKL ON)
        message(STATUS "MKL detected in BLAS libraries.")
    elseif(BLAS_LIBRARIES_LOWER MATCHES "openblas")
        set(VBCSR_HAS_OPENBLAS ON)
        message(STATUS "OpenBLAS detected.")
    else()
        set(VBCSR_HAS_BLAS ON)
        message(STATUS "Generic BLAS detected.")
    endif()
else()
    message(FATAL_ERROR "BLAS library not found.")
endif()

if(VBCSR_USE_ILP64)
    message(STATUS "Using 64-bit integers for BLAS interfaces.")
else()
    message(STATUS "Using 32-bit integers for BLAS interfaces.")
endif()

# Pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from git...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Source files
# Create the interface library for C++ core
add_library(vbcsr INTERFACE)
add_library(vbcsr::vbcsr ALIAS vbcsr)

# Link dependencies
target_link_libraries(vbcsr INTERFACE MPI::MPI_C MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

# Include directories
target_include_directories(vbcsr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vbcsr/core>
    $<INSTALL_INTERFACE:include>
)

# Compile definitions
if(VBCSR_HAS_MKL)
    target_compile_definitions(vbcsr INTERFACE VBCSR_USE_MKL)
endif()
if(VBCSR_HAS_OPENBLAS)
    target_compile_definitions(vbcsr INTERFACE VBCSR_USE_OPENBLAS)
endif()
if(VBCSR_HAS_BLAS)
    target_compile_definitions(vbcsr INTERFACE VBCSR_USE_BLAS)
endif()
if(VBCSR_USE_ILP64)
    target_compile_definitions(vbcsr INTERFACE VBCSR_USE_ILP64)
endif()

# Source files for Python binding
set(SOURCES
    vbcsr/pybind_vbcsr.cpp
)

# Create the extension module
pybind11_add_module(vbcsr_core ${SOURCES})

# Link libraries for Python binding
target_link_libraries(vbcsr_core PRIVATE vbcsr)

# Install rules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY vbcsr/core/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vbcsr)

# Install the library target
install(TARGETS vbcsr
    EXPORT vbcsrTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install the Python extension (optional, but good for pip install)
install(TARGETS vbcsr_core DESTINATION .)

# Export targets
install(EXPORT vbcsrTargets
    FILE vbcsrTargets.cmake
    NAMESPACE vbcsr::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vbcsr
)

# Create and install config files
set(CVERSION "0.1.0") # Set version as needed
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vbcsrConfigVersion.cmake"
    VERSION ${CVERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vbcsrConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vbcsrConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vbcsr
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vbcsrConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vbcsrConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vbcsr
)

# Tests
option(VBCSR_ENABLE_TESTS "Enable building tests" OFF)

if(VBCSR_ENABLE_TESTS)
    enable_testing()

    find_package(GTest)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, fetching from git...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        include(GoogleTest)
    endif()

    function(add_vbcsr_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE vbcsr GTest::gtest GTest::gtest_main Threads::Threads dl)
        add_test(NAME ${name} COMMAND mpirun -np 2 $<TARGET_FILE:${name}>)
    endfunction()

    add_vbcsr_test(test_pb_csr vbcsr/core/test/test_pb_csr.cpp)
    add_vbcsr_test(test_spmm vbcsr/core/test/test_spmm.cpp)
    add_vbcsr_test(test_block_csr vbcsr/core/test/test_block_csr.cpp)
    add_vbcsr_test(test_dist_graph vbcsr/core/test/test_dist_graph.cpp)
    add_vbcsr_test(test_robustness vbcsr/core/test/test_robustness.cpp)
    add_vbcsr_test(test_complex_dist_vector vbcsr/core/test/test_complex_dist_vector.cpp)
    # add_vbcsr_test(test_multi_sparsity vbcsr/core/test/test_multi_sparsity.cpp) # Fails with np=2
    add_vbcsr_test(test_backend_extensions vbcsr/core/test/test_backend_extensions.cpp)
    add_vbcsr_test(benchmark_dist vbcsr/core/test/benchmark_dist.cpp)
    add_vbcsr_test(benchmark_spmm_complex vbcsr/core/test/benchmark_spmm_complex.cpp)
    add_vbcsr_test(test_axpby vbcsr/core/test/test_axpby.cpp)
    add_vbcsr_test(test_axpby_diff_graph vbcsr/core/test/test_axpby_diff_graph.cpp)
    add_vbcsr_test(benchmark_axpby vbcsr/core/test/benchmark_axpby.cpp)
    add_vbcsr_test(test_density vbcsr/core/test/test_density.cpp)
    add_vbcsr_test(test_asymmetric_filter vbcsr/core/test/test_asymmetric_filter.cpp)
    add_vbcsr_test(test_block_arena vbcsr/core/test/test_block_arena.cpp)
    add_vbcsr_test(test_extract_batched vbcsr/core/test/test_extract_batched.cpp)
    add_vbcsr_test(test_extract_batched_extended vbcsr/core/test/test_extract_batched_extended.cpp)
    add_vbcsr_test(test_extract_batched_robust vbcsr/core/test/test_extract_batched_robust.cpp)
    add_vbcsr_test(test_hermitian_product vbcsr/core/test/test_hermitian_product.cpp)
    add_vbcsr_test(test_mult_graph_mismatch vbcsr/core/test/test_mult_graph_mismatch.cpp)
    add_vbcsr_test(test_subgraph vbcsr/core/test/test_subgraph.cpp)
    add_vbcsr_test(test_graphmf vbcsr/core/test/test_graphmf.cpp)
    add_vbcsr_test(test_dist_csr vbcsr/core/test/test_dist_csr.cpp)

    add_executable(test_block_csr_export vbcsr/core/test/test_block_csr_export.cpp)
    target_link_libraries(test_block_csr_export PRIVATE vbcsr GTest::gtest GTest::gtest_main Threads::Threads dl)
    add_test(NAME test_block_csr_export COMMAND mpirun -np 2 $<TARGET_FILE:test_block_csr_export>)
endif()
