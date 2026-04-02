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

# ParMETIS/METIS
find_library(PARMETIS_LIB NAMES parmetis HINTS $ENV{CONDA_PREFIX}/lib)
find_library(METIS_LIB NAMES metis HINTS $ENV{CONDA_PREFIX}/lib)
find_path(PARMETIS_INCLUDE_DIR NAMES parmetis.h HINTS $ENV{CONDA_PREFIX}/include)
find_path(METIS_INCLUDE_DIR NAMES metis.h HINTS $ENV{CONDA_PREFIX}/include)

if(PARMETIS_LIB AND METIS_LIB AND PARMETIS_INCLUDE_DIR AND METIS_INCLUDE_DIR)
    set(VBCSR_HAVE_PARMETIS ON)
    message(STATUS "Found ParMETIS: ${PARMETIS_LIB}")
    message(STATUS "Found METIS: ${METIS_LIB}")
    message(STATUS "Found ParMETIS includes: ${PARMETIS_INCLUDE_DIR}")
else()
    message(STATUS "ParMETIS or METIS not found. Will use Hilbert curve fallback.")
endif()

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
        # Prefer mkl_rt (Single Dynamic Library) over individual MKL libraries.
        # When multiple MKL consumers coexist (e.g. vbcsr + PARDISO wrapper),
        # linking separate libs (mkl_intel_lp64 + mkl_intel_thread + mkl_core)
        # causes PARDISO dispatcher conflicts. mkl_rt avoids this by providing
        # a single entry point with consistent internal state.
        find_library(MKL_RT_LIB NAMES mkl_rt HINTS $ENV{CONDA_PREFIX}/lib $ENV{MKLROOT}/lib)
        if(MKL_RT_LIB)
            message(STATUS "Swapping to MKL SDL (mkl_rt): ${MKL_RT_LIB}")
            set(BLAS_LIBRARIES ${MKL_RT_LIB})
        else()
            message(WARNING "mkl_rt not found, using individual MKL libraries. "
                            "This may cause conflicts with other MKL consumers (e.g. PARDISO).")
        endif()
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

include(CheckCXXSourceCompiles)

set(_vbcsr_saved_required_libs "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})

if(VBCSR_HAS_MKL OR VBCSR_HAS_OPENBLAS)
    check_cxx_source_compiles([=[
        extern "C" void cblas_dgemm_batch_strided(
            int, int, int, int, int, int, double,
            const double*, int, int,
            const double*, int, int,
            double, double*, int, int, int);
        int main() {
            cblas_dgemm_batch_strided(
                102, 111, 111,
                0, 0, 0,
                1.0,
                static_cast<const double*>(nullptr), 1, 0,
                static_cast<const double*>(nullptr), 1, 0,
                0.0,
                static_cast<double*>(nullptr), 1, 0, 0);
            return 0;
        }
    ]=] VBCSR_HAVE_CBLAS_DGEMM_BATCH_STRIDED)
endif()

if(VBCSR_HAS_MKL)
    check_cxx_source_compiles([=[
        extern "C" void cblas_dgemv_batch_strided(
            int, int, int, int, double,
            const double*, int, int,
            const double*, int, int,
            double, double*, int, int, int);
        int main() {
            cblas_dgemv_batch_strided(
                102, 111,
                0, 0,
                1.0,
                static_cast<const double*>(nullptr), 1, 0,
                static_cast<const double*>(nullptr), 1, 0,
                0.0,
                static_cast<double*>(nullptr), 1, 0, 0);
            return 0;
        }
    ]=] VBCSR_HAVE_CBLAS_DGEMV_BATCH_STRIDED)
endif()

set(CMAKE_REQUIRED_LIBRARIES "${_vbcsr_saved_required_libs}")
unset(_vbcsr_saved_required_libs)

set(VBCSR_SPARSE_VENDOR_INCLUDE_DIRS "")
set(VBCSR_SPARSE_VENDOR_LIBRARIES "")

if(VBCSR_USE_ILP64)
    message(STATUS "Skipping sparse vendor probes for ILP64 builds; CSR vendor sparse v1 currently assumes 32-bit sparse indices.")
else()
    if(VBCSR_HAS_MKL)
        set(VBCSR_MKL_SPARSE_HINTS
            $ENV{CONDA_PREFIX}/include
            $ENV{MKLROOT}/include
            /opt/intel/oneapi/mkl/latest/include
            /opt/intel/mkl/include)

        if(DEFINED ENV{CONDA_PREFIX})
            file(GLOB VBCSR_MKL_CONDA_INCLUDE_HINTS
                "$ENV{CONDA_PREFIX}/../../pkgs/mkl-include-*/include")
            list(APPEND VBCSR_MKL_SPARSE_HINTS ${VBCSR_MKL_CONDA_INCLUDE_HINTS})
        endif()

        find_path(VBCSR_MKL_SPARSE_INCLUDE_DIR
            NAMES mkl_spblas.h mkl.h
            HINTS ${VBCSR_MKL_SPARSE_HINTS})

        if(VBCSR_MKL_SPARSE_INCLUDE_DIR)
            set(_vbcsr_saved_required_libs "${CMAKE_REQUIRED_LIBRARIES}")
            set(_vbcsr_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
            set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
            set(CMAKE_REQUIRED_INCLUDES ${VBCSR_MKL_SPARSE_INCLUDE_DIR})
            check_cxx_source_compiles([=[
                #include <mkl_spblas.h>
                int main() {
                    sparse_matrix_t A = nullptr;
                    MKL_INT rows_start[1] = {0};
                    MKL_INT rows_end[1] = {1};
                    MKL_INT cols[1] = {0};
                    double vals[1] = {1.0};
                    const sparse_status_t status = mkl_sparse_d_create_csr(
                        &A,
                        SPARSE_INDEX_BASE_ZERO,
                        1,
                        1,
                        rows_start,
                        rows_end,
                        cols,
                        vals);
                    if (A != nullptr) {
                        mkl_sparse_destroy(A);
                    }
                    return status == SPARSE_STATUS_SUCCESS ? 0 : 1;
                }
            ]=] VBCSR_HAVE_MKL_SPARSE)
            set(CMAKE_REQUIRED_LIBRARIES "${_vbcsr_saved_required_libs}")
            set(CMAKE_REQUIRED_INCLUDES "${_vbcsr_saved_required_includes}")
            unset(_vbcsr_saved_required_libs)
            unset(_vbcsr_saved_required_includes)

            if(VBCSR_HAVE_MKL_SPARSE)
                list(APPEND VBCSR_SPARSE_VENDOR_INCLUDE_DIRS ${VBCSR_MKL_SPARSE_INCLUDE_DIR})
                message(STATUS "MKL sparse CSR backend enabled.")
            endif()
        endif()
    endif()

    if(NOT VBCSR_HAVE_MKL_SPARSE)
        find_path(VBCSR_AOCL_SPARSE_INCLUDE_DIR
            NAMES aoclsparse.h
            HINTS
                $ENV{CONDA_PREFIX}/include
                $ENV{AOCL_ROOT}/include
                $ENV{AOCLROOT}/include
                /opt/AMD/aocl-sparse/include
                /opt/aocl/include)
        find_library(VBCSR_AOCL_SPARSE_LIBRARY
            NAMES aoclsparse
            HINTS
                $ENV{CONDA_PREFIX}/lib
                $ENV{AOCL_ROOT}/lib
                $ENV{AOCLROOT}/lib
                /opt/AMD/aocl-sparse/lib
                /opt/aocl/lib)

        if(VBCSR_AOCL_SPARSE_INCLUDE_DIR AND VBCSR_AOCL_SPARSE_LIBRARY)
            set(_vbcsr_saved_required_libs "${CMAKE_REQUIRED_LIBRARIES}")
            set(_vbcsr_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
            set(CMAKE_REQUIRED_LIBRARIES ${VBCSR_AOCL_SPARSE_LIBRARY} ${BLAS_LIBRARIES})
            set(CMAKE_REQUIRED_INCLUDES ${VBCSR_AOCL_SPARSE_INCLUDE_DIR})
            check_cxx_source_compiles([=[
                #include <aoclsparse.h>
                int main() {
                    aoclsparse_matrix A = nullptr;
                    aoclsparse_int row_ptr[2] = {0, 1};
                    aoclsparse_int cols[1] = {0};
                    double vals[1] = {1.0};
                    const aoclsparse_status status = aoclsparse_create_dcsr(
                        &A,
                        aoclsparse_index_base_zero,
                        1,
                        1,
                        1,
                        row_ptr,
                        cols,
                        vals);
                    if (A != nullptr) {
                        aoclsparse_destroy(A);
                    }
                    return status == aoclsparse_status_success ? 0 : 1;
                }
            ]=] VBCSR_HAVE_AOCL_SPARSE)
            set(CMAKE_REQUIRED_LIBRARIES "${_vbcsr_saved_required_libs}")
            set(CMAKE_REQUIRED_INCLUDES "${_vbcsr_saved_required_includes}")
            unset(_vbcsr_saved_required_libs)
            unset(_vbcsr_saved_required_includes)

            if(VBCSR_HAVE_AOCL_SPARSE)
                list(APPEND VBCSR_SPARSE_VENDOR_INCLUDE_DIRS ${VBCSR_AOCL_SPARSE_INCLUDE_DIR})
                list(APPEND VBCSR_SPARSE_VENDOR_LIBRARIES ${VBCSR_AOCL_SPARSE_LIBRARY})
                message(STATUS "AOCL-Sparse CSR backend enabled.")
            endif()
        endif()
    endif()
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
target_link_libraries(vbcsr INTERFACE MPI::MPI_C MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_libraries(vbcsr INTERFACE $<BUILD_INTERFACE:${BLAS_LIBRARIES}>)
if(TARGET BLAS::BLAS)
    target_link_libraries(vbcsr INTERFACE $<INSTALL_INTERFACE:BLAS::BLAS>)
else()
    target_link_libraries(vbcsr INTERFACE $<INSTALL_INTERFACE:${BLAS_LIBRARIES}>)
endif()
if(VBCSR_SPARSE_VENDOR_LIBRARIES)
    target_link_libraries(vbcsr INTERFACE $<BUILD_INTERFACE:${VBCSR_SPARSE_VENDOR_LIBRARIES}>)
endif()
if(VBCSR_HAVE_PARMETIS)
    target_link_libraries(vbcsr INTERFACE ${PARMETIS_LIB} ${METIS_LIB})
    target_include_directories(vbcsr INTERFACE ${PARMETIS_INCLUDE_DIR} ${METIS_INCLUDE_DIR})
    target_compile_definitions(vbcsr INTERFACE VBCSR_HAVE_PARMETIS)
endif()

# Include directories
target_include_directories(vbcsr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vbcsr/core>
    $<INSTALL_INTERFACE:include>
)
if(VBCSR_SPARSE_VENDOR_INCLUDE_DIRS)
    target_include_directories(vbcsr INTERFACE $<BUILD_INTERFACE:${VBCSR_SPARSE_VENDOR_INCLUDE_DIRS}>)
endif()

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
if(VBCSR_HAVE_CBLAS_DGEMM_BATCH_STRIDED)
    target_compile_definitions(vbcsr INTERFACE VBCSR_BLAS_HAS_BATCH_GEMM)
endif()
if(VBCSR_HAVE_CBLAS_DGEMV_BATCH_STRIDED)
    target_compile_definitions(vbcsr INTERFACE VBCSR_BLAS_HAS_BATCH_GEMV)
endif()
if(VBCSR_HAVE_MKL_SPARSE)
    target_compile_definitions(vbcsr INTERFACE $<BUILD_INTERFACE:VBCSR_HAVE_MKL_SPARSE>)
endif()
if(VBCSR_HAVE_AOCL_SPARSE)
    target_compile_definitions(vbcsr INTERFACE $<BUILD_INTERFACE:VBCSR_HAVE_AOCL_SPARSE>)
endif()

# Source files for Python binding
set(SOURCES
    vbcsr/pybind_vbcsr.cpp
    vbcsr/pybind_atomic.cpp
)

# Create the extension module
pybind11_add_module(vbcsr_core ${SOURCES})

# Link libraries for Python binding
target_link_libraries(vbcsr_core PRIVATE vbcsr)
set_target_properties(vbcsr_core PROPERTIES
    BUILD_WITH_INSTALL_RPATH OFF
    INSTALL_RPATH ""
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
    SKIP_BUILD_RPATH TRUE
)

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
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}/lib" VBCSR_PYTHON_ENV_RPATH)
    install(CODE "
        if(EXISTS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/./$<TARGET_FILE_NAME:vbcsr_core>\" AND
           NOT IS_SYMLINK \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/./$<TARGET_FILE_NAME:vbcsr_core>\")
          file(RPATH_CHANGE
               FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/./$<TARGET_FILE_NAME:vbcsr_core>\"
               OLD_RPATH \"${VBCSR_PYTHON_ENV_RPATH}\"
               NEW_RPATH \"\")
        endif()
    ")
endif()

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
    include(CMakeParseArguments)

    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, fetching from git...")
        include(FetchContent)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    include(GoogleTest)

    function(add_vbcsr_gtest name source)
        set(options)
        set(oneValueArgs MPI_RANKS)
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE vbcsr GTest::gtest GTest::gtest_main Threads::Threads dl)
        if(ARG_MPI_RANKS)
            add_test(NAME ${name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_MPI_RANKS} $<TARGET_FILE:${name}>)
        else()
            add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
        endif()
    endfunction()

    function(add_vbcsr_mpi_test name source)
        set(options)
        set(oneValueArgs MPI_RANKS)
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
        if(NOT ARG_MPI_RANKS)
            set(ARG_MPI_RANKS 2)
        endif()
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE vbcsr Threads::Threads dl)
        add_test(NAME ${name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_MPI_RANKS} $<TARGET_FILE:${name}>)
    endfunction()

    function(add_vbcsr_plain_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE vbcsr Threads::Threads dl)
        add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
    endfunction()

    function(add_vbcsr_benchmark name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE vbcsr Threads::Threads dl)
    endfunction()

    add_vbcsr_mpi_test(test_pb_csr vbcsr/core/test/test_pb_csr.cpp)
    add_vbcsr_mpi_test(test_spmm vbcsr/core/test/test_spmm.cpp)
    add_vbcsr_mpi_test(test_block_csr vbcsr/core/test/test_block_csr.cpp)
    add_vbcsr_mpi_test(test_dist_graph vbcsr/core/test/test_dist_graph.cpp)
    add_vbcsr_mpi_test(test_robustness vbcsr/core/test/test_robustness.cpp)
    add_vbcsr_mpi_test(test_complex_dist_vector vbcsr/core/test/test_complex_dist_vector.cpp)
    add_vbcsr_mpi_test(test_backend_extensions vbcsr/core/test/test_backend_extensions.cpp)
    add_vbcsr_benchmark(benchmark_dist vbcsr/core/test/benchmark_dist.cpp)
    add_vbcsr_benchmark(benchmark_csr_vendor vbcsr/core/test/benchmark_csr_vendor.cpp)
    add_vbcsr_benchmark(benchmark_spmm_complex vbcsr/core/test/benchmark_spmm_complex.cpp)
    add_vbcsr_benchmark(benchmark_vbcsr_packed vbcsr/core/test/benchmark_vbcsr_packed.cpp)
    add_vbcsr_mpi_test(test_axpby vbcsr/core/test/test_axpby.cpp)
    add_vbcsr_mpi_test(test_axpby_diff_graph vbcsr/core/test/test_axpby_diff_graph.cpp)
    add_vbcsr_benchmark(benchmark_axpby vbcsr/core/test/benchmark_axpby.cpp)
    add_vbcsr_mpi_test(test_density vbcsr/core/test/test_density.cpp)
    add_vbcsr_mpi_test(test_asymmetric_filter vbcsr/core/test/test_asymmetric_filter.cpp)
    add_vbcsr_mpi_test(test_extract_batched vbcsr/core/test/test_extract_batched.cpp)
    add_vbcsr_mpi_test(test_extract_batched_extended vbcsr/core/test/test_extract_batched_extended.cpp)
    add_vbcsr_mpi_test(test_extract_batched_robust vbcsr/core/test/test_extract_batched_robust.cpp)
    add_vbcsr_mpi_test(test_hermitian_product vbcsr/core/test/test_hermitian_product.cpp)
    add_vbcsr_mpi_test(test_mult_graph_mismatch vbcsr/core/test/test_mult_graph_mismatch.cpp)
    add_vbcsr_mpi_test(test_subgraph vbcsr/core/test/test_subgraph.cpp)
    add_vbcsr_mpi_test(test_graphmf vbcsr/core/test/test_graphmf.cpp)
    add_vbcsr_mpi_test(test_dist_csr vbcsr/core/test/test_dist_csr.cpp)
    add_vbcsr_gtest(test_block_csr_export vbcsr/core/test/test_block_csr_export.cpp MPI_RANKS 2)

    add_vbcsr_gtest(test_atomic_data vbcsr/core/atomic/test/test_atomic_data.cpp MPI_RANKS 2)
    add_vbcsr_mpi_test(test_image_container vbcsr/core/atomic/test/test_image_container.cpp MPI_RANKS 2)
    add_vbcsr_plain_test(test_neighbourlist vbcsr/core/atomic/test/test_neighbourlist.cpp)
    add_vbcsr_gtest(test_neighbourlist_repro vbcsr/core/atomic/test/test_neighbourlist_repro.cpp)

    if(PARMETIS_LIB AND METIS_LIB AND PARMETIS_INCLUDE_DIR AND METIS_INCLUDE_DIR)
        foreach(target
            test_atomic_data
            test_image_container
        )
            target_link_libraries(${target} PRIVATE ${PARMETIS_LIB} ${METIS_LIB})
            target_include_directories(${target} PRIVATE ${PARMETIS_INCLUDE_DIR} ${METIS_INCLUDE_DIR})
        endforeach()
    endif()
endif()
