#ifndef VBCSR_DETAIL_BACKEND_BSR_VENDOR_CACHE_HPP
#define VBCSR_DETAIL_BACKEND_BSR_VENDOR_CACHE_HPP

#include "vendor_common.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace vbcsr::detail {

enum class BSRVendorBackendKind {
    None,
    MKL
};

inline const char* bsr_vendor_backend_name(BSRVendorBackendKind kind) {
    switch (kind) {
    case BSRVendorBackendKind::MKL:
        return "mkl";
    case BSRVendorBackendKind::None:
    default:
        return "none";
    }
}

template <typename T>
constexpr BSRVendorBackendKind preferred_bsr_vendor_backend() {
    constexpr bool supported_type =
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>;
    if constexpr (!supported_type) {
        return BSRVendorBackendKind::None;
    }
#if defined(VBCSR_HAVE_MKL_BSR_SPARSE)
    return BSRVendorBackendKind::MKL;
#else
    return BSRVendorBackendKind::None;
#endif
}

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
struct BSRVendorMKLTag {};
using BSRVendorMKLMMVariant = SparseVendorMKLMMVariant<BSRVendorMKLTag>;
using BSRVendorMKLHandle = SparseVendorMKLHandle<BSRVendorMKLMMVariant>;
#endif

template <typename T>
struct BSRVendorBatchEntry {
    // Owning storage for batch.row_block_offsets. Each batch corresponds to a
    // page with row-aware metadata.
    std::vector<int> row_block_offsets_storage;
    BSRPageBatch<const T> batch;

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    std::vector<MKL_INT> mm_rows_start_one;
    std::vector<MKL_INT> mm_rows_end_one;
    std::vector<MKL_INT> mm_cols_one;
    BSRVendorMKLHandle mkl;
#endif

    BSRVendorBatchEntry() = default;
    BSRVendorBatchEntry(const BSRVendorBatchEntry&) = delete;
    BSRVendorBatchEntry& operator=(const BSRVendorBatchEntry&) = delete;

    BSRVendorBatchEntry(BSRVendorBatchEntry&& other) noexcept
        : row_block_offsets_storage(std::move(other.row_block_offsets_storage)),
          batch(other.batch)
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
          , mm_rows_start_one(std::move(other.mm_rows_start_one)),
          mm_rows_end_one(std::move(other.mm_rows_end_one)),
          mm_cols_one(std::move(other.mm_cols_one)),
          mkl(std::move(other.mkl))
#endif
    {
        batch.row_block_offsets =
            row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
    }

    BSRVendorBatchEntry& operator=(BSRVendorBatchEntry&& other) noexcept {
        if (this != &other) {
            row_block_offsets_storage = std::move(other.row_block_offsets_storage);
            batch = other.batch;
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
            mm_rows_start_one = std::move(other.mm_rows_start_one);
            mm_rows_end_one = std::move(other.mm_rows_end_one);
            mm_cols_one = std::move(other.mm_cols_one);
            mkl = std::move(other.mkl);
#endif
            batch.row_block_offsets =
                row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
        }
        return *this;
    }

    void clear_vendor_handle() {
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
        mkl.reset();
#endif
    }
};

template <typename T>
struct BSRVendorCache {
    BSRVendorBackendKind kind = BSRVendorBackendKind::None;
    int num_block_cols = 0;
    std::vector<BSRVendorBatchEntry<T>> batches;

    void clear_vendor_handles() {
        for (auto& batch : batches) {
            batch.clear_vendor_handle();
        }
    }
};

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
template <typename T>
bool build_bsr_mkl_mv_handle(
    BSRVendorMKLHandle& handle,
    const BSRVendorBatchEntry<T>& entry,
    int num_block_cols) {
    const auto& batch = entry.batch;
    if (batch.block_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        batch.row_count() > std::numeric_limits<int>::max() ||
        num_block_cols > std::numeric_limits<int>::max()) {
        return false;
    }

    sparse_matrix_t raw_handle = nullptr;
    destroy_mkl_sparse_handle(raw_handle);

    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    const MKL_INT rows = static_cast<MKL_INT>(batch.row_count());
    const MKL_INT cols = static_cast<MKL_INT>(num_block_cols);
    const MKL_INT mkl_block_size = static_cast<MKL_INT>(batch.block_size);
    auto* row_begin = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.row_block_offsets));
    auto* row_end = row_begin + 1;
    auto* col_index = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.cols));

    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_bsr(
            &raw_handle,
            SPARSE_INDEX_BASE_ZERO,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            rows,
            cols,
            mkl_block_size,
            row_begin,
            row_end,
            col_index,
            const_cast<double*>(batch.values));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        status = mkl_sparse_z_create_bsr(
            &raw_handle,
            SPARSE_INDEX_BASE_ZERO,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            rows,
            cols,
            mkl_block_size,
            row_begin,
            row_end,
            col_index,
            reinterpret_cast<MKL_Complex16*>(
                const_cast<std::complex<double>*>(batch.values)));
    } else {
        return false;
    }

    if (status != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
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
bool build_bsr_mkl_mm_variant(
    BSRVendorMKLMMVariant& variant,
    const BSRVendorBatchEntry<T>& entry,
    int num_block_cols,
    int num_rhs) {
    const auto& batch = entry.batch;
    if (batch.block_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        batch.row_count() > std::numeric_limits<int>::max() ||
        num_block_cols > std::numeric_limits<int>::max()) {
        return false;
    }

    sparse_matrix_t raw_handle = nullptr;
    destroy_mkl_sparse_handle(raw_handle);

    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    const MKL_INT rows = static_cast<MKL_INT>(batch.row_count());
    const MKL_INT cols = static_cast<MKL_INT>(num_block_cols);
    const MKL_INT mkl_block_size = static_cast<MKL_INT>(batch.block_size);
    auto* row_begin = const_cast<MKL_INT*>(entry.mm_rows_start_one.data());
    auto* row_end = const_cast<MKL_INT*>(entry.mm_rows_end_one.data());
    auto* col_index = const_cast<MKL_INT*>(entry.mm_cols_one.data());

    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_bsr(
            &raw_handle,
            SPARSE_INDEX_BASE_ONE,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            rows,
            cols,
            mkl_block_size,
            row_begin,
            row_end,
            col_index,
            const_cast<double*>(batch.values));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        status = mkl_sparse_z_create_bsr(
            &raw_handle,
            SPARSE_INDEX_BASE_ONE,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            rows,
            cols,
            mkl_block_size,
            row_begin,
            row_end,
            col_index,
            reinterpret_cast<MKL_Complex16*>(
                const_cast<std::complex<double>*>(batch.values)));
    } else {
        return false;
    }

    if (status != SPARSE_STATUS_SUCCESS) {
        destroy_mkl_sparse_handle(raw_handle);
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
bool ensure_bsr_mkl_mm_handles(BSRVendorCache<T>& cache, int num_rhs) {
    for (auto& entry : cache.batches) {
        if (!ensure_mkl_mm_variant_with_lru(
                entry.mkl,
                num_rhs,
                [&](BSRVendorMKLMMVariant& variant) {
                    return build_bsr_mkl_mm_variant(
                        variant,
                        entry,
                        cache.num_block_cols,
                        num_rhs);
                })) {
            return false;
        }
    }
    return true;
}
#endif

template <typename T>
void build_bsr_vendor_cache(
    BSRVendorCache<T>& cache,
    const BSRApplyPlan<T>& plan,
    int num_block_cols) {
    cache.kind = preferred_bsr_vendor_backend<T>();
    cache.num_block_cols = num_block_cols;
    cache.batches.clear();
    cache.batches.reserve(plan.batches.size());

    for (const auto& apply_entry : plan.batches) {
        if (apply_entry.batch.block_count == 0) {
            continue;
        }

        BSRVendorBatchEntry<T> vendor_entry;
        vendor_entry.batch = apply_entry.batch;
        vendor_entry.row_block_offsets_storage.assign(
            apply_entry.batch.row_block_offsets,
            apply_entry.batch.row_block_offsets +
                static_cast<ptrdiff_t>(apply_entry.batch.row_count() + 1));
        vendor_entry.batch.row_block_offsets =
            vendor_entry.row_block_offsets_storage.data();
        cache.batches.push_back(std::move(vendor_entry));
    }

    if (cache.kind != BSRVendorBackendKind::None) {
        for (auto& entry : cache.batches) {
            bool ok = false;
            switch (cache.kind) {
            case BSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
                entry.mm_rows_start_one.resize(
                    static_cast<size_t>(entry.batch.row_count()));
                entry.mm_rows_end_one.resize(
                    static_cast<size_t>(entry.batch.row_count()));
                for (int row = 0; row < entry.batch.row_count(); ++row) {
                    entry.mm_rows_start_one[static_cast<size_t>(row)] = static_cast<MKL_INT>(
                        entry.batch.row_block_offsets[static_cast<size_t>(row)] + 1);
                    entry.mm_rows_end_one[static_cast<size_t>(row)] = static_cast<MKL_INT>(
                        entry.batch.row_block_offsets[static_cast<size_t>(row + 1)] + 1);
                }

                entry.mm_cols_one.resize(static_cast<size_t>(entry.batch.block_count));
                for (uint32_t block_index = 0;
                     block_index < entry.batch.block_count;
                     ++block_index) {
                    entry.mm_cols_one[static_cast<size_t>(block_index)] =
                        static_cast<MKL_INT>(entry.batch.cols[block_index] + 1);
                }
                ok = build_bsr_mkl_mv_handle(entry.mkl, entry, num_block_cols);
#endif
                break;
            case BSRVendorBackendKind::None:
            default:
                ok = true;
                break;
            }
            if (!ok) {
                cache.kind = BSRVendorBackendKind::None;
                cache.clear_vendor_handles();
                break;
            }
        }
    }
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_BSR_VENDOR_CACHE_HPP
