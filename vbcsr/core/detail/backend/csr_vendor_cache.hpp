#ifndef VBCSR_DETAIL_BACKEND_CSR_VENDOR_CACHE_HPP
#define VBCSR_DETAIL_BACKEND_CSR_VENDOR_CACHE_HPP

#include "backend_common.hpp"
#include "vendor_common.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef VBCSR_HAVE_AOCL_SPARSE
#include <aoclsparse.h>
#endif

namespace vbcsr::detail {

enum class CSRVendorBackendKind {
    None,
    MKL,
    AOCL
};

inline const char* csr_vendor_backend_name(CSRVendorBackendKind kind) {
    switch (kind) {
    case CSRVendorBackendKind::MKL:
        return "mkl";
    case CSRVendorBackendKind::AOCL:
        return "aocl";
    case CSRVendorBackendKind::None:
    default:
        return "none";
    }
}

template <typename T>
constexpr CSRVendorBackendKind preferred_csr_vendor_backend() {
    constexpr bool supported_type =
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>;
    if constexpr (!supported_type) {
        return CSRVendorBackendKind::None;
    }
#if defined(VBCSR_HAVE_MKL_SPARSE)
    return CSRVendorBackendKind::MKL;
#elif defined(VBCSR_HAVE_AOCL_SPARSE)
    return CSRVendorBackendKind::AOCL;
#else
    return CSRVendorBackendKind::None;
#endif
}

#ifdef VBCSR_HAVE_MKL_SPARSE
struct CSRVendorMKLTag {};
using CSRVendorMKLMMVariant = SparseVendorMKLMMVariant<CSRVendorMKLTag>;
using CSRVendorMKLHandle = SparseVendorMKLHandle<CSRVendorMKLMMVariant>;
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
struct CSRVendorAOCLHandle {
    aoclsparse_matrix handle = nullptr;

    CSRVendorAOCLHandle() = default;
    CSRVendorAOCLHandle(const CSRVendorAOCLHandle&) = delete;
    CSRVendorAOCLHandle& operator=(const CSRVendorAOCLHandle&) = delete;

    CSRVendorAOCLHandle(CSRVendorAOCLHandle&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    CSRVendorAOCLHandle& operator=(CSRVendorAOCLHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    ~CSRVendorAOCLHandle() {
        reset();
    }

    void reset() {
        if (handle != nullptr) {
            aoclsparse_destroy(&handle);
            handle = nullptr;
        }
    }
};

struct CSRVendorAOCLDescr {
    aoclsparse_mat_descr handle = nullptr;

    CSRVendorAOCLDescr() = default;
    CSRVendorAOCLDescr(const CSRVendorAOCLDescr&) = delete;
    CSRVendorAOCLDescr& operator=(const CSRVendorAOCLDescr&) = delete;

    CSRVendorAOCLDescr(CSRVendorAOCLDescr&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    CSRVendorAOCLDescr& operator=(CSRVendorAOCLDescr&& other) noexcept {
        if (this != &other) {
            reset();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    ~CSRVendorAOCLDescr() {
        reset();
    }

    void reset() {
        if (handle != nullptr) {
            aoclsparse_destroy_mat_descr(handle);
            handle = nullptr;
        }
    }
};
#endif

template <typename T>
struct CSRVendorPageEntry {
    // Owning storage for batch.row_offsets.
    std::vector<int> row_offsets_storage;
    CSRPageBatch<const T> batch;

#ifdef VBCSR_HAVE_MKL_SPARSE
    CSRVendorMKLHandle mkl;
#endif
#ifdef VBCSR_HAVE_AOCL_SPARSE
    CSRVendorAOCLHandle aocl;
#endif

    CSRVendorPageEntry() = default;
    CSRVendorPageEntry(const CSRVendorPageEntry&) = delete;
    CSRVendorPageEntry& operator=(const CSRVendorPageEntry&) = delete;

    CSRVendorPageEntry(CSRVendorPageEntry&& other) noexcept
        : row_offsets_storage(std::move(other.row_offsets_storage)),
          batch(other.batch)
#ifdef VBCSR_HAVE_MKL_SPARSE
          , mkl(std::move(other.mkl))
#endif
#ifdef VBCSR_HAVE_AOCL_SPARSE
          , aocl(std::move(other.aocl))
#endif
    {
        batch.row_offsets = row_offsets_storage.empty() ? nullptr : row_offsets_storage.data();
    }

    CSRVendorPageEntry& operator=(CSRVendorPageEntry&& other) noexcept {
        if (this != &other) {
            row_offsets_storage = std::move(other.row_offsets_storage);
            batch = other.batch;
#ifdef VBCSR_HAVE_MKL_SPARSE
            mkl = std::move(other.mkl);
#endif
#ifdef VBCSR_HAVE_AOCL_SPARSE
            aocl = std::move(other.aocl);
#endif
            batch.row_offsets =
                row_offsets_storage.empty() ? nullptr : row_offsets_storage.data();
        }
        return *this;
    }

    void clear_vendor_handle() {
#ifdef VBCSR_HAVE_MKL_SPARSE
        mkl.reset();
#endif
#ifdef VBCSR_HAVE_AOCL_SPARSE
        aocl.reset();
#endif
    }
};

template <typename T>
struct CSRVendorCache {
    CSRVendorBackendKind kind = CSRVendorBackendKind::None;
    int num_cols = 0;
    std::vector<CSRVendorPageEntry<T>> pages;

#ifdef VBCSR_HAVE_AOCL_SPARSE
    CSRVendorAOCLDescr aocl_descr;
#endif

    void clear_vendor_handles() {
        for (auto& page : pages) {
            page.clear_vendor_handle();
        }
#ifdef VBCSR_HAVE_AOCL_SPARSE
        aocl_descr.reset();
#endif
    }
};

template <typename T>
CSRVendorPageEntry<T> build_csr_vendor_page_entry(
    const std::vector<int>& row_ptr,
    CSRNnzSlice<const T> page_slice) {
    CSRVendorPageEntry<T> entry;
    entry.batch.cols = page_slice.cols;
    entry.batch.values = page_slice.values;
    entry.batch.nnz_count = page_slice.nnz_count;
    entry.batch.page_index = page_slice.page_index;
    entry.batch.first_nnz = page_slice.first_nnz;
    if (page_slice.nnz_count == 0) {
        return entry;
    }

    const int begin = static_cast<int>(page_slice.first_nnz);
    const int end = begin + static_cast<int>(page_slice.nnz_count);
    const PageRowSpan row_span = find_page_row_span(row_ptr, begin, end);
    entry.batch.row_begin = row_span.row_begin;
    entry.batch.row_end = row_span.row_end;
    entry.row_offsets_storage.reserve(static_cast<size_t>(row_span.row_count() + 1));
    emit_page_local_row_ptr(row_ptr, begin, end, row_span, [&](int page_local_offset) {
        entry.row_offsets_storage.push_back(page_local_offset);
    });
    entry.batch.row_offsets = entry.row_offsets_storage.data();
    return entry;
}

#ifdef VBCSR_HAVE_MKL_SPARSE
template <typename T>
bool build_csr_mkl_raw_handle(
    sparse_matrix_t& out_handle,
    CSRPageBatch<const T> batch,
    int num_cols) {
    destroy_mkl_sparse_handle(out_handle);

    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    const MKL_INT rows = static_cast<MKL_INT>(batch.row_count());
    const MKL_INT cols_count = static_cast<MKL_INT>(num_cols);
    auto* row_begin = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.row_offsets));
    auto* row_end = row_begin + 1;
    auto* col_idx = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.cols));

    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_csr(
            &out_handle,
            SPARSE_INDEX_BASE_ZERO,
            rows,
            cols_count,
            row_begin,
            row_end,
            col_idx,
            const_cast<double*>(batch.values));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        status = mkl_sparse_z_create_csr(
            &out_handle,
            SPARSE_INDEX_BASE_ZERO,
            rows,
            cols_count,
            row_begin,
            row_end,
            col_idx,
            reinterpret_cast<MKL_Complex16*>(
                const_cast<std::complex<double>*>(batch.values)));
    } else {
        return false;
    }

    if (status != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(out_handle);
        return false;
    }
    return true;
}

template <typename T>
bool build_csr_mkl_mv_handle(
    CSRVendorMKLHandle& handle,
    CSRPageBatch<const T> batch,
    int num_cols) {
    sparse_matrix_t raw_handle = nullptr;
    if (!build_csr_mkl_raw_handle(raw_handle, batch, num_cols)) {
        return false;
    }

    const matrix_descr descr = make_mkl_descr();
    if (mkl_sparse_set_mv_hint(
            raw_handle,
            SPARSE_OPERATION_NON_TRANSPOSE,
            descr,
            1) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }
    if (mkl_sparse_set_mv_hint(
            raw_handle,
            mkl_adjoint_operation<T>(),
            descr,
            1) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }
    if (mkl_sparse_optimize(raw_handle) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }

    destroy_mkl_sparse_handle(handle.mv_handle);
    handle.mv_handle = raw_handle;
    return true;
}

template <typename T>
bool build_csr_mkl_mm_variant(
    CSRVendorMKLMMVariant& variant,
    CSRPageBatch<const T> batch,
    int num_cols,
    int num_rhs) {
    sparse_matrix_t raw_handle = nullptr;
    if (!build_csr_mkl_raw_handle(raw_handle, batch, num_cols)) {
        return false;
    }

    const matrix_descr descr = make_mkl_descr();
    if (mkl_sparse_set_mm_hint(
            raw_handle,
            SPARSE_OPERATION_NON_TRANSPOSE,
            descr,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            static_cast<MKL_INT>(num_rhs),
            1) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }
    if (mkl_sparse_set_mm_hint(
            raw_handle,
            mkl_adjoint_operation<T>(),
            descr,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            static_cast<MKL_INT>(num_rhs),
            1) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }
    if (mkl_sparse_optimize(raw_handle) != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
        return false;
    }

    variant.reset();
    variant.num_rhs = num_rhs;
    variant.handle = raw_handle;
    return true;
}

template <typename T>
bool ensure_csr_mkl_mm_handles(CSRVendorCache<T>& cache, int num_rhs) {
    for (auto& entry : cache.pages) {
        if (!ensure_mkl_mm_variant_with_lru(
                entry.mkl,
                num_rhs,
                [&](CSRVendorMKLMMVariant& variant) {
                    return build_csr_mkl_mm_variant(
                        variant,
                        entry.batch,
                        cache.num_cols,
                        num_rhs);
                })) {
            return false;
        }
    }
    return true;
}
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
template <typename T>
bool build_csr_aocl_page_handle(
    CSRVendorAOCLHandle& handle,
    CSRPageBatch<const T> batch,
    aoclsparse_mat_descr descr,
    int num_cols) {
    aoclsparse_status status = aoclsparse_status_not_implemented;
    const aoclsparse_int rows = static_cast<aoclsparse_int>(batch.row_count());
    const aoclsparse_int cols_count = static_cast<aoclsparse_int>(num_cols);
    const aoclsparse_int nnz = static_cast<aoclsparse_int>(batch.nnz_count);
    auto* row_ptr_local = reinterpret_cast<aoclsparse_int*>(const_cast<int*>(batch.row_offsets));
    auto* col_idx = reinterpret_cast<aoclsparse_int*>(const_cast<int*>(batch.cols));

    if constexpr (std::is_same_v<T, double>) {
        status = aoclsparse_create_dcsr(
            &handle.handle,
            aoclsparse_index_base_zero,
            rows,
            cols_count,
            nnz,
            row_ptr_local,
            col_idx,
            const_cast<double*>(batch.values));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        status = aoclsparse_create_zcsr(
            &handle.handle,
            aoclsparse_index_base_zero,
            rows,
            cols_count,
            nnz,
            row_ptr_local,
            col_idx,
            reinterpret_cast<aoclsparse_double_complex*>(
                const_cast<std::complex<double>*>(batch.values)));
    } else {
        return false;
    }

    if (status != aoclsparse_status_success) {
        handle.reset();
        return false;
    }

    if constexpr (std::is_same_v<T, double>) {
        aoclsparse_set_mv_hint(
            handle.handle,
            aoclsparse_operation_none,
            descr,
            1);
        aoclsparse_set_mv_hint(
            handle.handle,
            aoclsparse_operation_transpose,
            descr,
            1);
    }

    aoclsparse_set_mm_hint(handle.handle, aoclsparse_operation_none, descr, 1);
    if constexpr (std::is_same_v<T, double>) {
        aoclsparse_set_mm_hint(
            handle.handle,
            aoclsparse_operation_transpose,
            descr,
            1);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        aoclsparse_set_mm_hint(
            handle.handle,
            aoclsparse_operation_conjugate_transpose,
            descr,
            1);
    }
    return aoclsparse_optimize(handle.handle) == aoclsparse_status_success;
}
#endif

template <typename T, typename PageFn>
void build_csr_vendor_cache(
    CSRVendorCache<T>& cache,
    const std::vector<int>& row_ptr,
    uint32_t page_count,
    int num_cols,
    PageFn&& page_for_index) {
    cache.kind = preferred_csr_vendor_backend<T>();
    cache.num_cols = num_cols;
    cache.pages.clear();
    cache.pages.reserve(page_count);

    for (uint32_t page_index = 0; page_index < page_count; ++page_index) {
        const auto page_slice = page_for_index(page_index);
        if (page_slice.nnz_count == 0) {
            continue;
        }
        cache.pages.push_back(build_csr_vendor_page_entry(row_ptr, page_slice));
    }

    if (cache.kind == CSRVendorBackendKind::None) {
        return;
    }

#ifdef VBCSR_HAVE_AOCL_SPARSE
    if (cache.kind == CSRVendorBackendKind::AOCL) {
        if (aoclsparse_create_mat_descr(&cache.aocl_descr.handle) != aoclsparse_status_success) {
            cache.kind = CSRVendorBackendKind::None;
            cache.clear_vendor_handles();
            return;
        }
        aoclsparse_set_mat_type(cache.aocl_descr.handle, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_index_base(cache.aocl_descr.handle, aoclsparse_index_base_zero);
    }
#endif

    for (auto& entry : cache.pages) {
        bool ok = false;
        switch (cache.kind) {
        case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
            ok = build_csr_mkl_mv_handle(entry.mkl, entry.batch, num_cols);
#endif
            break;
        case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
            ok = build_csr_aocl_page_handle(
                entry.aocl,
                entry.batch,
                cache.aocl_descr.handle,
                num_cols);
#endif
            break;
        case CSRVendorBackendKind::None:
        default:
            ok = true;
            break;
        }
        if (!ok) {
            cache.kind = CSRVendorBackendKind::None;
            cache.clear_vendor_handles();
            return;
        }
    }
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_CSR_VENDOR_CACHE_HPP
