cmake_minimum_required(VERSION 3.15)
project(vbcsr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find MKL first, then BLAS
find_package(MKL)
if(MKL_FOUND)
    message(STATUS "MKL found: ${MKL_INCLUDE_DIRS}")
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARY_DIRS})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
else()
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        add_definitions(-DUSE_BLAS)
    else()
        message(WARNING "No BLAS library found. Performance may be degraded.")
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
target_link_libraries(vbcsr INTERFACE MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

# Include directories
target_include_directories(vbcsr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vbcsr/core>
    $<INSTALL_INTERFACE:include>
)

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
