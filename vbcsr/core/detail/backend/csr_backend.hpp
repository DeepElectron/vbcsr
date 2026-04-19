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
            vendor_cache = std::make_unique<CSRVendorCache<T>>();
            build_vendor_cache_locked(*vendor_cache, row_ptr, col_ind, num_cols);
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

        for (auto& entry : vendor_cache->pages) {
            if (!ensure_mkl_mm_variant_locked(
                    entry,
                    vendor_cache->num_cols,
                    num_rhs)) {
                return false;
            }
        }
        return true;
    }
#endif

private:
    void invalidate_vendor_cache() {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        vendor_cache.reset();
        vendor_launch_count.store(0, std::memory_order_release);
    }

    static CSRVendorPageEntry<T> build_vendor_page_entry(
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
        // A storage page can begin or end in the middle of a CSR row, so vendor
        // batches need a row window plus row_ptr values rebased into page-local
        // NNZ offsets.
        const RebasedRowSpan row_span = find_rebased_row_span(row_ptr, begin, end);
        entry.batch.row_begin = row_span.row_begin;
        entry.batch.row_end = row_span.row_end;
        entry.row_offsets_storage.reserve(static_cast<size_t>(row_span.row_count() + 1));
        emit_rebased_row_offsets(row_ptr, begin, end, row_span, [&](int rebased_offset) {
            entry.row_offsets_storage.push_back(rebased_offset);
        });
        entry.batch.row_offsets = entry.row_offsets_storage.data();
        return entry;
    }

#ifdef VBCSR_HAVE_MKL_SPARSE
    bool build_mkl_raw_handle(
        sparse_matrix_t& out_handle,
        CSRPageBatch<const T> batch,
        int num_cols) const {
        destroy_mkl_sparse_handle(out_handle);

        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        const MKL_INT rows = static_cast<MKL_INT>(batch.row_count());
        const MKL_INT cols_count = static_cast<MKL_INT>(num_cols);
        auto* row_begin = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.row_offsets));
        // MKL expects CSR row_begin/row_end arrays of length rows. Our compact page-local
        // row_ptr has length rows + 1, so row_end is just row_begin shifted by one entry.
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

    bool build_mkl_mv_handle(
        CSRVendorMKLHandle& handle,
        CSRPageBatch<const T> batch,
        int num_cols) const {
        sparse_matrix_t raw_handle = nullptr;
        if (!build_mkl_raw_handle(raw_handle, batch, num_cols)) {
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
        CSRVendorMKLMMVariant& variant,
        CSRPageBatch<const T> batch,
        int num_cols,
        int num_rhs) const {
        sparse_matrix_t raw_handle = nullptr;
        if (!build_mkl_raw_handle(raw_handle, batch, num_cols)) {
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

    bool ensure_mkl_mm_variant_locked(
        CSRVendorPageEntry<T>& entry,
        int num_cols,
        int num_rhs) const {
        return ensure_mkl_mm_variant_with_lru(
            entry.mkl,
            num_rhs,
            [&](CSRVendorMKLMMVariant& variant) {
                return build_mkl_mm_variant(
                    variant,
                    entry.batch,
                    num_cols,
                    num_rhs);
            });
    }
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
    bool build_aocl_page_handle(
        CSRVendorAOCLHandle& handle,
        CSRPageBatch<const T> batch,
        aoclsparse_mat_descr descr,
        int num_cols) const {
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

    void build_vendor_cache_locked(
        CSRVendorCache<T>& cache,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        int num_cols) const {
        cache.kind = preferred_csr_vendor_backend<T>();
        cache.num_cols = num_cols;
        cache.pages.clear();
        cache.pages.reserve(values.page_count());

        for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
            const auto page_slice = page(col_ind, page_index);
            if (page_slice.nnz_count == 0) {
                continue;
            }
            cache.pages.push_back(build_vendor_page_entry(row_ptr, page_slice));
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
                ok = build_mkl_mv_handle(entry.mkl, entry.batch, num_cols);
#endif
                break;
            case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
                ok = build_aocl_page_handle(
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
