#ifndef VBCSR_DETAIL_BACKEND_MATRIX_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_MATRIX_BACKEND_HPP

#include "bsr_backend.hpp"
#include "csr_backend.hpp"
#include "vbcsr_backend.hpp"
#include <stdexcept>
#include <variant>

namespace vbcsr::detail {

template <typename T, typename Kernel>
using MatrixBackendHandle = std::variant<
    std::monostate,
    CSRMatrixBackend<T>,
    BSRMatrixBackend<T>,
    VBCSRMatrixBackend<T, Kernel>>;

template <typename Backend, typename T, typename Kernel>
Backend& require_backend(
    MatrixBackendHandle<T, Kernel>& handle,
    const char* error_message) {
    auto* storage = std::get_if<Backend>(&handle);
    if (storage == nullptr) {
        throw std::logic_error(error_message);
    }
    return *storage;
}

template <typename Backend, typename T, typename Kernel>
const Backend& require_backend(
    const MatrixBackendHandle<T, Kernel>& handle,
    const char* error_message) {
    auto* storage = std::get_if<Backend>(&handle);
    if (storage == nullptr) {
        throw std::logic_error(error_message);
    }
    return *storage;
}

template <typename T, typename Kernel>
CSRMatrixBackend<T>& require_csr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<CSRMatrixBackend<T>>(
        handle,
        "Active backend is not the CSR storage path");
}

template <typename T, typename Kernel>
const CSRMatrixBackend<T>& require_csr_backend(
    const MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<CSRMatrixBackend<T>>(
        handle,
        "Active backend is not the CSR storage path");
}

template <typename T, typename Kernel>
BSRMatrixBackend<T>& require_bsr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<BSRMatrixBackend<T>>(
        handle,
        "Active backend is not the BSR storage path");
}

template <typename T, typename Kernel>
const BSRMatrixBackend<T>& require_bsr_backend(
    const MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<BSRMatrixBackend<T>>(
        handle,
        "Active backend is not the BSR storage path");
}

template <typename T, typename Kernel>
VBCSRMatrixBackend<T, Kernel>& require_vbcsr_backend(
    MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<VBCSRMatrixBackend<T, Kernel>>(
        handle,
        "Active backend is not the VBCSR storage path");
}

template <typename T, typename Kernel>
const VBCSRMatrixBackend<T, Kernel>& require_vbcsr_backend(
    const MatrixBackendHandle<T, Kernel>& handle) {
    return require_backend<VBCSRMatrixBackend<T, Kernel>>(
        handle,
        "Active backend is not the VBCSR storage path");
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_MATRIX_BACKEND_HPP
