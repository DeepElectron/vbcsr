#ifndef VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP

#include "backend_common.hpp"

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

template <typename T>
struct BSRBlockSlice {
    const int* cols = nullptr;
    T* values = nullptr;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t block_value_count = 0;
    uint32_t page_index = 0;
    uint64_t first_block = 0;
};

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
struct BSRVendorMKLTag {};
using BSRVendorMKLMMVariant = SparseVendorMKLMMVariant<BSRVendorMKLTag>;
using BSRVendorMKLHandle = SparseVendorMKLHandle<BSRVendorMKLMMVariant>;
#endif

template <typename T>
struct BSRPageBatch {
    const int* cols = nullptr;
    T* values = nullptr;
    const int* row_block_offsets = nullptr;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t block_value_count = 0;
    uint32_t page_index = 0;
    uint64_t first_block = 0;
    int row_begin = 0;
    int row_end = 0;

    int row_count() const {
        return row_end - row_begin;
    }

    uint32_t row_block_start(int row) const {
        return static_cast<uint32_t>(
            row_block_offsets[static_cast<size_t>(row - row_begin)]);
    }

    uint32_t row_block_end(int row) const {
        return static_cast<uint32_t>(
            row_block_offsets[static_cast<size_t>(row - row_begin + 1)]);
    }

    const T* block_ptr(uint32_t local_block_index) const {
        return values + static_cast<size_t>(local_block_index) * block_value_count;
    }

    T* block_ptr(uint32_t local_block_index) {
        return values + static_cast<size_t>(local_block_index) * block_value_count;
    }
};

template <typename T>
struct BSRApplyBatchEntry {
    // Owning storage for batch.row_block_offsets.
    std::vector<int> row_block_offsets_storage;
    BSRPageBatch<const T> batch;

    BSRApplyBatchEntry() = default;
    BSRApplyBatchEntry(const BSRApplyBatchEntry&) = delete;
    BSRApplyBatchEntry& operator=(const BSRApplyBatchEntry&) = delete;

    BSRApplyBatchEntry(BSRApplyBatchEntry&& other) noexcept
        : row_block_offsets_storage(std::move(other.row_block_offsets_storage)),
          batch(other.batch) {
        batch.row_block_offsets =
            row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
    }

    BSRApplyBatchEntry& operator=(BSRApplyBatchEntry&& other) noexcept {
        if (this != &other) {
            row_block_offsets_storage = std::move(other.row_block_offsets_storage);
            batch = other.batch;
            batch.row_block_offsets =
                row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
        }
        return *this;
    }
};

template <typename T>
struct BSRApplyPlan {
    std::vector<BSRApplyBatchEntry<T>> batches;
};

template <typename T>
struct BSRVendorBatchEntry {
    // Owning storage for batch.row_block_offsets.
    // each batch correspond to a page, with row aware metadata
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

template <typename T>
struct BSRMatrixBackend {
    static uint32_t max_blocks_per_page(int uniform_block_size) {
        if (uniform_block_size <= 0) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint64_t block_values =
            static_cast<uint64_t>(uniform_block_size) * static_cast<uint64_t>(uniform_block_size);
        const uint64_t index_limit =
            static_cast<uint64_t>(std::numeric_limits<int>::max());
        const uint64_t by_values =
            block_values == 0
                ? 1u
                : static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) /
                    block_values;
        const uint64_t bounded =
            std::max<uint64_t>(1, std::min<uint64_t>(by_values, index_limit));
        return static_cast<uint32_t>(bounded);
    }

    static uint32_t normalize_blocks_per_page(
        uint32_t requested,
        int uniform_block_size) {
        if (requested == 0) {
            return max_blocks_per_page(uniform_block_size);
        }
        if (uniform_block_size <= 0) {
            return std::max<uint32_t>(requested, 1u);
        }
        return static_cast<uint32_t>(std::clamp<uint64_t>(
            requested,
            1u,
            static_cast<uint64_t>(max_blocks_per_page(uniform_block_size))));
    }

    int block_size = 0;
    PagedBuffer<T> values;
    mutable std::unique_ptr<BSRApplyPlan<T>> apply_plan;
    mutable std::mutex apply_plan_mutex;
    mutable std::unique_ptr<BSRVendorCache<T>> vendor_cache;
    mutable std::mutex vendor_cache_mutex;
    mutable std::atomic<uint64_t> vendor_launch_count{0};
    uint32_t configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();

    BSRMatrixBackend() = default;

    BSRMatrixBackend(int uniform_block_size, uint32_t blocks_per_page)
        : block_size(uniform_block_size),
          configured_blocks_per_page_(
              normalize_blocks_per_page(blocks_per_page, uniform_block_size)),
          values(std::max<uint32_t>(
              configured_blocks_per_page_ *
                  static_cast<uint32_t>(
                      uniform_block_size * uniform_block_size),
              1u)) {}

    BSRMatrixBackend(const BSRMatrixBackend&) = delete;
    BSRMatrixBackend& operator=(const BSRMatrixBackend&) = delete;

    BSRMatrixBackend(BSRMatrixBackend&& other) noexcept
        : block_size(other.block_size),
          values(std::move(other.values)),
          configured_blocks_per_page_(other.configured_blocks_per_page_) {
        other.block_size = 0;
        other.configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();
        other.vendor_launch_count.store(0, std::memory_order_release);
    }

    BSRMatrixBackend& operator=(BSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            block_size = other.block_size;
            values = std::move(other.values);
            configured_blocks_per_page_ = other.configured_blocks_per_page_;
            invalidate_apply_plan();
            vendor_launch_count.store(0, std::memory_order_release);
            other.block_size = 0;
            other.configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();
            other.vendor_launch_count.store(0, std::memory_order_release);
        }
        return *this;
    }

    uint32_t configured_blocks_per_page() const {
        return configured_blocks_per_page_;
    }

    uint32_t active_blocks_per_page() const {
        return static_cast<uint32_t>(
            values.elements_per_page() / std::max<size_t>(values_per_block(), 1));
    }

    uint64_t scalar_value_count() const {
        return values.size();
    }

    size_t values_per_block() const {
        return static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
    }

    size_t block_count() const {
        const size_t values_in_block = values_per_block();
        return values_in_block == 0
            ? 0
            : static_cast<size_t>(values.size()) / values_in_block;
    }

    void initialize_structure(uint64_t logical_blocks, int uniform_block_size) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(configured_blocks_per_page_, block_size);
        const uint32_t blocks_per_page = logical_blocks == 0
            ? configured_blocks_per_page_
            : static_cast<uint32_t>(
                  std::min<uint64_t>(logical_blocks, configured_blocks_per_page_));
        values = PagedBuffer<T>(std::max<uint32_t>(
            blocks_per_page * static_cast<uint32_t>(values_per_block()),
            1u));
        invalidate_apply_plan();
        values.resize(
            logical_blocks * static_cast<uint64_t>(values_per_block()));
    }

    void initialize_structure(
        uint64_t logical_blocks,
        int uniform_block_size,
        uint32_t blocks_per_page) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(blocks_per_page, uniform_block_size);
        invalidate_apply_plan();
        initialize_structure(logical_blocks, uniform_block_size);
    }

    T* block_ptr(int slot) {
        return values.element_ptr(
            static_cast<uint64_t>(slot) *
            static_cast<uint64_t>(values_per_block()));
    }

    const T* block_ptr(int slot) const {
        return values.element_ptr(
            static_cast<uint64_t>(slot) *
            static_cast<uint64_t>(values_per_block()));
    }

    BSRBlockSlice<T> page(const std::vector<int>& col_ind, uint32_t page_index) {
        auto value_page = values.page(page_index);
        const uint32_t block_value_count =
            static_cast<uint32_t>(this->values_per_block());
        const uint64_t first_block =
            value_page.first_element / static_cast<uint64_t>(block_value_count);
        const uint32_t block_count = value_page.count / block_value_count;
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRBlockSlice<T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            block_value_count,
            page_index,
            first_block};
    }

    BSRBlockSlice<const T> page(
        const std::vector<int>& col_ind,
        uint32_t page_index) const {
        auto value_page = values.page(page_index);
        const uint32_t block_value_count =
            static_cast<uint32_t>(this->values_per_block());
        const uint64_t first_block =
            value_page.first_element / static_cast<uint64_t>(block_value_count);
        const uint32_t block_count = value_page.count / block_value_count;
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRBlockSlice<const T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            block_value_count,
            page_index,
            first_block};
    }

    const BSRApplyPlan<T>& ensure_apply_plan(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (apply_plan == nullptr) {
            auto plan = std::make_unique<BSRApplyPlan<T>>();
            plan->batches.reserve(values.page_count());

            for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
                const auto page_slice = page(col_ind, page_index);
                if (page_slice.block_count == 0) {
                    continue;
                }

                BSRApplyBatchEntry<T> batch_entry;
                batch_entry.batch.cols = page_slice.cols;
                batch_entry.batch.values = page_slice.values;
                batch_entry.batch.block_count = page_slice.block_count;
                batch_entry.batch.block_size = page_slice.block_size;
                batch_entry.batch.block_value_count = page_slice.block_value_count;
                batch_entry.batch.page_index = page_slice.page_index;
                batch_entry.batch.first_block = page_slice.first_block;

                const int begin = static_cast<int>(page_slice.first_block);
                const int end = begin + static_cast<int>(page_slice.block_count);
                const RebasedRowSpan row_span = find_rebased_row_span(row_ptr, begin, end);
                batch_entry.batch.row_begin = row_span.row_begin;
                batch_entry.batch.row_end = row_span.row_end;
                batch_entry.row_block_offsets_storage.reserve(
                    static_cast<size_t>(row_span.row_count() + 1));
                emit_rebased_row_offsets(
                    row_ptr,
                    begin,
                    end,
                    row_span,
                    [&](int rebased_offset) {
                        batch_entry.row_block_offsets_storage.push_back(rebased_offset);
                    });
                batch_entry.batch.row_block_offsets =
                    batch_entry.row_block_offsets_storage.data();
                plan->batches.push_back(std::move(batch_entry));
            }

            apply_plan = std::move(plan);
        }
        return *apply_plan;
    }

    const BSRVendorCache<T>& ensure_vendor_cache(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        int num_block_cols) const {
        const auto& plan = ensure_apply_plan(row_ptr, col_ind);
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache == nullptr) {
            auto cache = std::make_unique<BSRVendorCache<T>>();
            cache->kind = preferred_bsr_vendor_backend<T>();
            cache->num_block_cols = num_block_cols;
            cache->batches.reserve(plan.batches.size());

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
                cache->batches.push_back(std::move(vendor_entry));
            }

            if (cache->kind != BSRVendorBackendKind::None) {
                for (auto& entry : cache->batches) {
                    bool ok = false;
                    switch (cache->kind) {
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
                        ok = build_mkl_mv_handle(entry.mkl, entry, num_block_cols);
#endif
                        break;
                    case BSRVendorBackendKind::None:
                    default:
                        ok = true;
                        break;
                    }
                    if (!ok) {
                        cache->kind = BSRVendorBackendKind::None;
                        cache->clear_vendor_handles();
                        break;
                    }
                }
            }

            vendor_cache = std::move(cache);
        }
        return *vendor_cache;
    }

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    bool ensure_mkl_mm_handles(
        const BSRVendorCache<T>& cache,
        int num_rhs) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (cache.kind != BSRVendorBackendKind::MKL ||
            vendor_cache == nullptr ||
            vendor_cache.get() != &cache) {
            return false;
        }

        for (auto& entry : vendor_cache->batches) {
            if (!ensure_mkl_mm_variant_with_lru(
                    entry.mkl,
                    num_rhs,
                    [&](BSRVendorMKLMMVariant& variant) {
                        return build_mkl_mm_variant(
                            variant,
                            entry,
                            vendor_cache->num_block_cols,
                            num_rhs);
                    })) {
                return false;
            }
        }
        return true;
    }
#endif

private:
    void invalidate_apply_plan() const {
        {
            std::lock_guard<std::mutex> lock(apply_plan_mutex);
            apply_plan.reset();
        }
        invalidate_vendor_cache();
    }

    void invalidate_vendor_cache() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        vendor_cache.reset();
        vendor_launch_count.store(0, std::memory_order_release);
    }

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    bool build_mkl_mv_handle(
        BSRVendorMKLHandle& handle,
        const BSRVendorBatchEntry<T>& entry,
        int num_block_cols) const {
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

    bool build_mkl_mm_variant(
        BSRVendorMKLMMVariant& variant,
        const BSRVendorBatchEntry<T>& entry,
        int num_block_cols,
        int num_rhs) const {
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

#endif

public:
    // Inspection and instrumentation helpers used by tests, benchmarks, and
    // vendor dispatch counters. They are kept at the end of the class so the
    // storage/build logic stays easy to read top-to-bottom.
    BSRVendorBackendKind vendor_backend_kind() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache != nullptr) {
            return vendor_cache->kind;
        }
        return preferred_bsr_vendor_backend<T>();
    }

    std::string vendor_backend_name() const {
        return bsr_vendor_backend_name(vendor_backend_kind());
    }

    const void* vendor_cache_identity() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        return vendor_cache.get();
    }

    uint64_t get_vendor_launch_count() const {
        return vendor_launch_count.load(std::memory_order_acquire);
    }

    void reset_vendor_launch_count() const {
        vendor_launch_count.store(0, std::memory_order_release);
    }

    void note_vendor_launch(uint64_t batch_calls = 1) const {
        vendor_launch_count.fetch_add(batch_calls, std::memory_order_acq_rel);
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP
