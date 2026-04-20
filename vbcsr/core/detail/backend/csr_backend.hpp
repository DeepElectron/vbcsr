#ifndef VBCSR_DETAIL_BACKEND_CSR_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_CSR_BACKEND_HPP

#include "backend_common.hpp"

namespace vbcsr::detail {

// Raw NNZ window into the CSR structure/value arrays.
// This is an internal storage slice: row ownership comes from the caller.
template <typename T>
struct CSRNnzSlice {
    const int* cols = nullptr;
    T* values = nullptr;
    uint32_t nnz_count = 0;
    uint32_t page_index = 0;
    uint64_t first_nnz = 0;
};

// Execution-facing CSR page view with row ownership and page-local row offsets.
template <typename T>
struct CSRPageBatch {
    const int* cols = nullptr;
    T* values = nullptr;
    // Page-local CSR row pointer for this batch's NNZ window.
    // Length is row_count() + 1.
    // Each entry is rebased to this batch, so row_offsets[0] is always 0 and
    // row_offsets[row_count()] is nnz_count.
    // Row r in [row_begin, row_end) owns local NNZ
    // [row_offsets[r - row_begin], row_offsets[r - row_begin + 1]).
    const int* row_offsets = nullptr;
    uint32_t nnz_count = 0;
    uint32_t page_index = 0;
    uint64_t first_nnz = 0;
    int row_begin = 0;
    int row_end = 0;

    int row_count() const {
        return row_end - row_begin;
    }
};

} // namespace vbcsr::detail

#include "csr_vendor_cache.hpp"

namespace vbcsr::detail {

template <typename T>
struct CSRMatrixBackend {
    static constexpr uint32_t max_page_size() {
        constexpr uint64_t hard_limit =
            std::min<uint64_t>(
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                static_cast<uint64_t>(std::numeric_limits<int>::max()));
        return static_cast<uint32_t>(hard_limit);
    }

    static uint32_t normalize_page_size(uint32_t requested) {
        if (requested == 0) {
            return max_page_size();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(requested, 1u, static_cast<uint64_t>(max_page_size())));
    }

    PagedBuffer<T> values;
    // Lazily-built vendor cache used by benchmarks and vendor dispatch paths.
    mutable std::unique_ptr<CSRVendorCache<T>> vendor_cache;
    mutable std::mutex vendor_cache_mutex;
    mutable std::atomic<uint64_t> vendor_launch_count{0};
    // Requested page size limit; the live buffer may use a smaller page when nnz_count is smaller.
    uint32_t configured_page_size_ = max_page_size();

    CSRMatrixBackend()
        : values(max_page_size()) {}

    explicit CSRMatrixBackend(uint32_t page_size)
        : values(normalize_page_size(page_size)),
          configured_page_size_(normalize_page_size(page_size)) {}

    CSRMatrixBackend(const CSRMatrixBackend&) = delete;
    CSRMatrixBackend& operator=(const CSRMatrixBackend&) = delete;

    CSRMatrixBackend(CSRMatrixBackend&& other) noexcept
        : values(std::move(other.values)),
          configured_page_size_(other.configured_page_size_) {
        other.vendor_launch_count.store(0, std::memory_order_release);
        other.configured_page_size_ = max_page_size();
    }

    CSRMatrixBackend& operator=(CSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            values = std::move(other.values);
            configured_page_size_ = other.configured_page_size_;
            invalidate_vendor_cache();
            vendor_launch_count.store(0, std::memory_order_release);
            other.vendor_launch_count.store(0, std::memory_order_release);
            other.configured_page_size_ = max_page_size();
        }
        return *this;
    }

    uint32_t configured_page_size() const {
        return configured_page_size_;
    }

    uint32_t active_page_size() const {
        return values.elements_per_page();
    }

    uint32_t page_count() const {
        return values.page_count();
    }

    uint64_t nnz_count() const {
        return values.size();
    }

    void initialize_structure(uint64_t logical_nnz) {
        const uint32_t page_size = logical_nnz == 0
            ? configured_page_size_
            : static_cast<uint32_t>(std::min<uint64_t>(logical_nnz, configured_page_size_));
        values = PagedBuffer<T>(page_size);
        invalidate_vendor_cache();
        values.resize(logical_nnz);
    }

    void initialize_structure(uint64_t logical_nnz, uint32_t page_size) {
        configured_page_size_ = normalize_page_size(page_size);
        initialize_structure(logical_nnz);
    }

    T* value_ptr(int slot) {
        return values.element_ptr(static_cast<uint64_t>(slot));
    }

    const T* value_ptr(int slot) const {
        return values.element_ptr(static_cast<uint64_t>(slot));
    }

    CSRNnzSlice<T> page(const std::vector<int>& col_ind, uint32_t page_index) {
        auto value_page = values.page(page_index);
        if (value_page.first_element + value_page.count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("CSRMatrixBackend::page column span out of bounds");
        }
        return CSRNnzSlice<T>{
            col_ind.data() + value_page.first_element,
            value_page.data,
            value_page.count,
            page_index,
            value_page.first_element
        };
    }

    CSRNnzSlice<const T> page(const std::vector<int>& col_ind, uint32_t page_index) const {
        auto value_page = values.page(page_index);
        if (value_page.first_element + value_page.count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("CSRMatrixBackend::page column span out of bounds");
        }
        return CSRNnzSlice<const T>{
            col_ind.data() + value_page.first_element,
            value_page.data,
            value_page.count,
            page_index,
            value_page.first_element
        };
    }

    template <typename Fn>
    void for_each_row_slice(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        int row,
        Fn&& fn) const {
        uint64_t current = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]);
        const uint64_t end = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row) + 1]);
        const uint32_t page_capacity = values.elements_per_page();
        while (current < end) {
            const uint32_t page_index = static_cast<uint32_t>(current / page_capacity);
            const uint32_t local_offset = static_cast<uint32_t>(current % page_capacity);
            const auto page_slice = page(col_ind, page_index);
            const uint32_t nnz_count = static_cast<uint32_t>(
                std::min<uint64_t>(page_slice.nnz_count - local_offset, end - current));
            if (local_offset > page_slice.nnz_count || nnz_count > page_slice.nnz_count - local_offset) {
                throw std::out_of_range("CSRMatrixBackend::for_each_row_slice range out of bounds");
            }
            fn(CSRNnzSlice<const T>{
                page_slice.cols + local_offset,
                page_slice.values + local_offset,
                nnz_count,
                page_slice.page_index,
                page_slice.first_nnz + local_offset
            });
            current += nnz_count;
        }
    }

    // Vendor execution support used by kernels/csr_apply.hpp.
    const CSRVendorCache<T>& ensure_vendor_cache(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        int num_cols) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache == nullptr) {
            auto cache = std::make_unique<CSRVendorCache<T>>();
            // The backend owns cache lifetime and invalidation because vendor
            // handles point into this storage; descriptor construction itself
            // lives in the vendor helper.
            build_csr_vendor_cache(
                *cache,
                row_ptr,
                values.page_count(),
                num_cols,
                [&](uint32_t page_index) {
                    return page(col_ind, page_index);
                });
            vendor_cache = std::move(cache);
        }
        return *vendor_cache;
    }

    void note_vendor_launch(uint64_t page_calls = 1) const {
        vendor_launch_count.fetch_add(page_calls, std::memory_order_acq_rel);
    }

#ifdef VBCSR_HAVE_MKL_SPARSE
    bool ensure_mkl_mm_handles(
        const CSRVendorCache<T>& cache,
        int num_rhs) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (cache.kind != CSRVendorBackendKind::MKL ||
            vendor_cache == nullptr ||
            vendor_cache.get() != &cache) {
            return false;
        }

        // Keep the backend as the synchronization owner while delegating MKL
        // MM-handle materialization to the vendor cache helper.
        return ensure_csr_mkl_mm_handles(*vendor_cache, num_rhs);
    }
#endif

private:
    void invalidate_vendor_cache() {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        vendor_cache.reset();
        vendor_launch_count.store(0, std::memory_order_release);
    }

public:
    // Diagnostics and test hooks.
    const char* vendor_backend_name() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        const CSRVendorBackendKind kind =
            vendor_cache != nullptr ? vendor_cache->kind : preferred_csr_vendor_backend<T>();
        return csr_vendor_backend_name(kind);
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
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_CSR_BACKEND_HPP
