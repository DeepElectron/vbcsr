#ifndef VBCSR_DETAIL_BACKEND_HANDLE_HPP
#define VBCSR_DETAIL_BACKEND_HANDLE_HPP

#include "paged_array.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef VBCSR_HAVE_MKL_SPARSE
#ifdef VBCSR_USE_ILP64
#ifndef MKL_ILP64
#define MKL_ILP64
#endif
#endif
#include <mkl_spblas.h>
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
#include <aoclsparse.h>
#endif

namespace vbcsr::detail {

template <typename T>
class ConstBoundVector {
public:
    ConstBoundVector() = default;
    explicit ConstBoundVector(const std::vector<T>& storage) : storage_(&storage) {}

    void unbind() {
        storage_ = nullptr;
    }

    void bind(const std::vector<T>& storage) {
        storage_ = &storage;
    }

    size_t size() const {
        require_bound();
        return storage_->size();
    }

    bool empty() const {
        require_bound();
        return storage_->empty();
    }

    const T* data() const {
        require_bound();
        return storage_->data();
    }

    auto begin() const {
        require_bound();
        return storage_->begin();
    }

    auto end() const {
        require_bound();
        return storage_->end();
    }

    const T& operator[](size_t idx) const {
        require_bound();
        return (*storage_)[idx];
    }

    friend bool operator==(const ConstBoundVector& lhs, const ConstBoundVector& rhs) {
        return *lhs.storage_ == *rhs.storage_;
    }

    friend bool operator!=(const ConstBoundVector& lhs, const ConstBoundVector& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const ConstBoundVector& lhs, const std::vector<T>& rhs) {
        return *lhs.storage_ == rhs;
    }

    friend bool operator==(const std::vector<T>& lhs, const ConstBoundVector& rhs) {
        return lhs == *rhs.storage_;
    }

    friend bool operator!=(const ConstBoundVector& lhs, const std::vector<T>& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const std::vector<T>& lhs, const ConstBoundVector& rhs) {
        return !(lhs == rhs);
    }

    operator const std::vector<T>&() const {
        require_bound();
        return *storage_;
    }

private:
    void require_bound() const {
        if (storage_ == nullptr) {
            throw std::logic_error("ConstBoundVector is not attached to active backend storage");
        }
    }

    const std::vector<T>* storage_ = nullptr;
};

template <typename T>
struct CSRPageView {
    const int* cols = nullptr;
    T* vals = nullptr;
    uint32_t nnz = 0;
    uint32_t page_id = 0;
    uint64_t global_nnz_begin = 0;
};

template <typename T>
struct CSRRowSegment {
    int row = -1;
    uint32_t page_id = 0;
    uint64_t global_nnz_begin = 0;
    uint32_t local_offset = 0;
    uint32_t length = 0;
    bool is_row_start = false;
    bool is_row_end = false;
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

struct CSRVendorPageMetadata {
    uint32_t page_id = 0;
    uint64_t global_nnz_begin = 0;
    uint32_t nnz = 0;
    int row_begin = 0;
    int row_end = 0;
};

#ifdef VBCSR_HAVE_MKL_SPARSE
struct CSRVendorMKLHandle {
    sparse_matrix_t handle = nullptr;

    CSRVendorMKLHandle() = default;
    CSRVendorMKLHandle(const CSRVendorMKLHandle&) = delete;
    CSRVendorMKLHandle& operator=(const CSRVendorMKLHandle&) = delete;

    CSRVendorMKLHandle(CSRVendorMKLHandle&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    CSRVendorMKLHandle& operator=(CSRVendorMKLHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    ~CSRVendorMKLHandle() {
        reset();
    }

    void reset() {
        if (handle != nullptr) {
            mkl_sparse_destroy(handle);
            handle = nullptr;
        }
    }
};
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
            aoclsparse_destroy(handle);
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
    CSRVendorPageMetadata metadata;
    std::vector<int> row_ptr;

#ifdef VBCSR_HAVE_MKL_SPARSE
    CSRVendorMKLHandle mkl;
#endif
#ifdef VBCSR_HAVE_AOCL_SPARSE
    CSRVendorAOCLHandle aocl;
#endif

    int row_count() const {
        return metadata.row_end - metadata.row_begin;
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
#ifdef VBCSR_HAVE_MKL_SPARSE
    int prepared_mm_columns = -1;
#endif

    void clear_vendor_handles() {
        for (auto& page : pages) {
            page.clear_vendor_handle();
        }
#ifdef VBCSR_HAVE_AOCL_SPARSE
        aocl_descr.reset();
#endif
#ifdef VBCSR_HAVE_MKL_SPARSE
        prepared_mm_columns = -1;
#endif
    }
};

template <typename T>
struct BSRPageView {
    const int* cols = nullptr;
    T* vals = nullptr;
    uint32_t nblocks = 0;
    uint32_t bsz = 0;
    uint32_t block_elems = 0;
    uint32_t page_id = 0;
    uint64_t global_block_begin = 0;
};

template <typename T>
struct BSRRowSegment {
    int block_row = -1;
    uint32_t page_id = 0;
    uint64_t global_block_begin = 0;
    uint32_t local_block_offset = 0;
    uint32_t block_count = 0;
    bool is_row_start = false;
    bool is_row_end = false;
};

template <typename T>
struct CSRMatrixBackend {
    PagedArray<int> cols;
    PagedArray<T> values;
    mutable std::unique_ptr<CSRVendorCache<T>> vendor_cache;
    mutable std::mutex vendor_cache_mutex;
    mutable std::atomic<uint64_t> vendor_launch_count{0};

    CSRMatrixBackend() = default;

    explicit CSRMatrixBackend(uint32_t nnz_per_page)
        : cols(nnz_per_page), values(nnz_per_page) {}

    CSRMatrixBackend(const CSRMatrixBackend&) = delete;
    CSRMatrixBackend& operator=(const CSRMatrixBackend&) = delete;

    CSRMatrixBackend(CSRMatrixBackend&& other) noexcept
        : cols(std::move(other.cols)),
          values(std::move(other.values)) {
        other.vendor_launch_count.store(0, std::memory_order_release);
    }

    CSRMatrixBackend& operator=(CSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            cols = std::move(other.cols);
            values = std::move(other.values);
            invalidate_vendor_cache();
            vendor_launch_count.store(0, std::memory_order_release);
            other.vendor_launch_count.store(0, std::memory_order_release);
        }
        return *this;
    }

    static uint32_t default_nnz_per_page() {
        constexpr size_t kTargetBytes = 1u << 20;
        constexpr size_t elems = kTargetBytes / (sizeof(int) + sizeof(T));
        return static_cast<uint32_t>(std::max<size_t>(elems, 1));
    }

    void configure_page_capacity(uint32_t nnz_per_page = 0) {
        const uint32_t capacity = nnz_per_page == 0 ? default_nnz_per_page() : nnz_per_page;
        cols = PagedArray<int>(capacity);
        values = PagedArray<T>(capacity);
        invalidate_vendor_cache();
    }

    size_t local_scalar_nnz() const {
        return static_cast<size_t>(values.size());
    }

    size_t local_block_nnz() const {
        return static_cast<size_t>(values.size());
    }

    void initialize_structure(const std::vector<int>& col_ind, uint32_t nnz_per_page = 0) {
        configure_page_capacity(nnz_per_page);
        cols.resize(col_ind.size());
        values.resize(col_ind.size());
        for (uint64_t idx = 0; idx < static_cast<uint64_t>(col_ind.size()); ++idx) {
            cols[idx] = col_ind[static_cast<size_t>(idx)];
        }
    }

    int col_at(int slot) const {
        return cols[static_cast<uint64_t>(slot)];
    }

    T* value_ptr(int slot) {
        return values.ptr(static_cast<uint64_t>(slot));
    }

    const T* value_ptr(int slot) const {
        return values.ptr(static_cast<uint64_t>(slot));
    }

    const int* col_ptr(int slot) const {
        return cols.ptr(static_cast<uint64_t>(slot));
    }

    CSRPageView<T> page_view(uint32_t page_id) {
        auto col_span = cols.page_span(page_id);
        auto val_span = values.page_span(page_id);
        return CSRPageView<T>{
            col_span.data,
            val_span.data,
            std::min(col_span.length, val_span.length),
            page_id,
            col_span.global_begin};
    }

    CSRPageView<const T> page_view(uint32_t page_id) const {
        auto col_span = cols.page_span(page_id);
        auto val_span = values.page_span(page_id);
        return CSRPageView<const T>{
            col_span.data,
            val_span.data,
            std::min(col_span.length, val_span.length),
            page_id,
            col_span.global_begin};
    }

    template <typename Fn>
    void for_each_page_view(Fn&& fn) {
        for (uint32_t page_id = 0; page_id < values.page_count(); ++page_id) {
            fn(page_view(page_id));
        }
    }

    template <typename Fn>
    void for_each_page_view(Fn&& fn) const {
        for (uint32_t page_id = 0; page_id < values.page_count(); ++page_id) {
            fn(page_view(page_id));
        }
    }

    template <typename Fn>
    void for_each_row_segment(const std::vector<int>& row_ptr, int row, Fn&& fn) const {
        uint64_t current = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]);
        const uint64_t end = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row) + 1]);
        const uint32_t page_capacity = values.page_capacity();
        while (current < end) {
            const uint32_t page_id = static_cast<uint32_t>(current / page_capacity);
            const uint32_t local_offset = static_cast<uint32_t>(current % page_capacity);
            const auto view = page_view(page_id);
            const uint32_t chunk = static_cast<uint32_t>(
                std::min<uint64_t>(view.nnz - local_offset, end - current));
            fn(
                CSRPageView<const T>{
                    view.cols + local_offset,
                    view.vals + local_offset,
                    chunk,
                    view.page_id,
                    current},
                CSRRowSegment<T>{
                    row,
                    view.page_id,
                    current,
                    local_offset,
                    chunk,
                    current == static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]),
                    current + chunk == end});
            current += chunk;
        }
    }

    const CSRVendorCache<T>& ensure_vendor_cache(const std::vector<int>& row_ptr, int num_cols) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache == nullptr) {
            vendor_cache = std::make_unique<CSRVendorCache<T>>();
            build_vendor_cache_locked(*vendor_cache, row_ptr, num_cols);
        }
        return *vendor_cache;
    }

    CSRVendorBackendKind vendor_backend_kind() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache != nullptr) {
            return vendor_cache->kind;
        }
        return preferred_csr_vendor_backend<T>();
    }

    std::string vendor_backend_name() const {
        return csr_vendor_backend_name(vendor_backend_kind());
    }

    uint64_t get_vendor_launch_count() const {
        return vendor_launch_count.load(std::memory_order_acquire);
    }

    void reset_vendor_launch_count() const {
        vendor_launch_count.store(0, std::memory_order_release);
    }

    void note_vendor_launch(uint64_t page_calls = 1) const {
        vendor_launch_count.fetch_add(page_calls, std::memory_order_acq_rel);
    }

private:
    void invalidate_vendor_cache() {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        vendor_cache.reset();
        vendor_launch_count.store(0, std::memory_order_release);
    }

    static CSRVendorPageEntry<T> build_vendor_page_entry(
        const std::vector<int>& row_ptr,
        CSRPageView<const T> page) {
        CSRVendorPageEntry<T> entry;
        entry.metadata.page_id = page.page_id;
        entry.metadata.global_nnz_begin = page.global_nnz_begin;
        entry.metadata.nnz = page.nnz;
        if (page.nnz == 0) {
            return entry;
        }

        const int begin = static_cast<int>(page.global_nnz_begin);
        const int end = begin + static_cast<int>(page.nnz);
        const auto row_begin_it =
            std::upper_bound(row_ptr.begin() + 1, row_ptr.end(), begin);
        const auto row_end_it =
            std::lower_bound(row_ptr.begin(), row_ptr.end() - 1, end);

        entry.metadata.row_begin = static_cast<int>(std::distance(row_ptr.begin(), row_begin_it)) - 1;
        entry.metadata.row_end = static_cast<int>(std::distance(row_ptr.begin(), row_end_it));

        entry.row_ptr.reserve(static_cast<size_t>(entry.metadata.row_end - entry.metadata.row_begin + 1));
        for (int row = entry.metadata.row_begin; row <= entry.metadata.row_end; ++row) {
            const int clamped = std::clamp(row_ptr[static_cast<size_t>(row)], begin, end);
            entry.row_ptr.push_back(clamped - begin);
        }
        return entry;
    }

#ifdef VBCSR_HAVE_MKL_SPARSE
    static matrix_descr make_mkl_descr() {
        matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.mode = SPARSE_FILL_MODE_FULL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        return descr;
    }

    bool build_mkl_page_handle(
        CSRVendorPageEntry<T>& entry,
        CSRPageView<const T> page,
        int num_cols) const {
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        const MKL_INT rows = static_cast<MKL_INT>(entry.row_count());
        const MKL_INT cols_count = static_cast<MKL_INT>(num_cols);
        auto* row_begin = reinterpret_cast<MKL_INT*>(entry.row_ptr.data());
        auto* row_end = row_begin + 1;
        auto* col_idx = reinterpret_cast<MKL_INT*>(const_cast<int*>(page.cols));

        if constexpr (std::is_same_v<T, double>) {
            status = mkl_sparse_d_create_csr(
                &entry.mkl.handle,
                SPARSE_INDEX_BASE_ZERO,
                rows,
                cols_count,
                row_begin,
                row_end,
                col_idx,
                const_cast<double*>(page.vals));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = mkl_sparse_z_create_csr(
                &entry.mkl.handle,
                SPARSE_INDEX_BASE_ZERO,
                rows,
                cols_count,
                row_begin,
                row_end,
                col_idx,
                reinterpret_cast<MKL_Complex16*>(const_cast<std::complex<double>*>(page.vals)));
        } else {
            return false;
        }

        if (status != SPARSE_STATUS_SUCCESS) {
            entry.mkl.reset();
            return false;
        }

        const matrix_descr descr = make_mkl_descr();
        mkl_sparse_set_mv_hint(entry.mkl.handle, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1);
        if constexpr (std::is_same_v<T, double>) {
            mkl_sparse_set_mv_hint(entry.mkl.handle, SPARSE_OPERATION_TRANSPOSE, descr, 1);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            mkl_sparse_set_mv_hint(entry.mkl.handle, SPARSE_OPERATION_CONJUGATE_TRANSPOSE, descr, 1);
        }
        return mkl_sparse_optimize(entry.mkl.handle) == SPARSE_STATUS_SUCCESS;
    }
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
    bool build_aocl_page_handle(
        CSRVendorPageEntry<T>& entry,
        CSRPageView<const T> page,
        aoclsparse_mat_descr descr,
        int num_cols) const {
        aoclsparse_status status = aoclsparse_status_not_implemented;
        const aoclsparse_int rows = static_cast<aoclsparse_int>(entry.row_count());
        const aoclsparse_int cols_count = static_cast<aoclsparse_int>(num_cols);
        const aoclsparse_int nnz = static_cast<aoclsparse_int>(page.nnz);
        auto* row_ptr_local = reinterpret_cast<aoclsparse_int*>(entry.row_ptr.data());
        auto* col_idx = reinterpret_cast<aoclsparse_int*>(const_cast<int*>(page.cols));

        if constexpr (std::is_same_v<T, double>) {
            status = aoclsparse_create_dcsr(
                &entry.aocl.handle,
                aoclsparse_index_base_zero,
                rows,
                cols_count,
                nnz,
                row_ptr_local,
                col_idx,
                const_cast<double*>(page.vals));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = aoclsparse_create_zcsr(
                &entry.aocl.handle,
                aoclsparse_index_base_zero,
                rows,
                cols_count,
                nnz,
                row_ptr_local,
                col_idx,
                reinterpret_cast<aoclsparse_double_complex*>(const_cast<std::complex<double>*>(page.vals)));
        } else {
            return false;
        }

        if (status != aoclsparse_status_success) {
            entry.aocl.reset();
            return false;
        }

        aoclsparse_set_mm_hint(entry.aocl.handle, aoclsparse_operation_none, descr, 1);
        if constexpr (std::is_same_v<T, double>) {
            aoclsparse_set_mm_hint(entry.aocl.handle, aoclsparse_operation_transpose, descr, 1);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            aoclsparse_set_mm_hint(
                entry.aocl.handle,
                aoclsparse_operation_conjugate_transpose,
                descr,
                1);
        }
        return aoclsparse_optimize(entry.aocl.handle) == aoclsparse_status_success;
    }
#endif

    void build_vendor_cache_locked(
        CSRVendorCache<T>& cache,
        const std::vector<int>& row_ptr,
        int num_cols) const {
        cache.kind = preferred_csr_vendor_backend<T>();
        cache.num_cols = num_cols;
        cache.pages.clear();
        cache.pages.reserve(values.page_count());

        for (uint32_t page_id = 0; page_id < values.page_count(); ++page_id) {
            const auto page = page_view(page_id);
            if (page.nnz == 0) {
                continue;
            }
            cache.pages.push_back(build_vendor_page_entry(row_ptr, page));
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
            const auto page = page_view(entry.metadata.page_id);
            bool ok = false;
            switch (cache.kind) {
            case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
                ok = build_mkl_page_handle(entry, page, num_cols);
#endif
                break;
            case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
                ok = build_aocl_page_handle(entry, page, cache.aocl_descr.handle, num_cols);
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
};

template <typename T>
struct BSRMatrixBackend {
    int block_size = 0;
    PagedArray<int> cols;
    PagedArray<T> values;

    BSRMatrixBackend() = default;

    BSRMatrixBackend(int uniform_block_size, uint32_t blocks_per_page)
        : block_size(uniform_block_size),
          cols(blocks_per_page),
          values(std::max<uint32_t>(blocks_per_page * static_cast<uint32_t>(uniform_block_size * uniform_block_size), 1u)) {}

    static uint32_t default_blocks_per_page(int uniform_block_size) {
        const size_t block_elems = static_cast<size_t>(uniform_block_size) * static_cast<size_t>(uniform_block_size);
        const size_t per_block_bytes = sizeof(int) + block_elems * sizeof(T);
        const size_t kTargetBytes = 1u << 20;
        const size_t elems = per_block_bytes == 0 ? 1 : kTargetBytes / per_block_bytes;
        return static_cast<uint32_t>(std::max<size_t>(elems, 1));
    }

    void configure_page_capacity(uint32_t blocks_per_page = 0) {
        const uint32_t capacity = blocks_per_page == 0 ? default_blocks_per_page(block_size) : blocks_per_page;
        cols = PagedArray<int>(capacity);
        values = PagedArray<T>(std::max<uint32_t>(capacity * static_cast<uint32_t>(block_elems()), 1u));
    }

    size_t local_scalar_nnz() const {
        return static_cast<size_t>(values.size());
    }

    size_t block_elems() const {
        return static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
    }

    size_t local_block_nnz() const {
        const size_t elems = block_elems();
        return elems == 0 ? 0 : static_cast<size_t>(values.size()) / elems;
    }

    void initialize_structure(const std::vector<int>& col_ind, int uniform_block_size, uint32_t blocks_per_page = 0) {
        block_size = uniform_block_size;
        configure_page_capacity(blocks_per_page);
        cols.resize(col_ind.size());
        values.resize(static_cast<uint64_t>(col_ind.size()) * static_cast<uint64_t>(block_elems()));
        for (uint64_t idx = 0; idx < static_cast<uint64_t>(col_ind.size()); ++idx) {
            cols[idx] = col_ind[static_cast<size_t>(idx)];
        }
    }

    int col_at(int slot) const {
        return cols[static_cast<uint64_t>(slot)];
    }

    T* block_ptr(int slot) {
        return values.ptr(static_cast<uint64_t>(slot) * static_cast<uint64_t>(block_elems()));
    }

    const T* block_ptr(int slot) const {
        return values.ptr(static_cast<uint64_t>(slot) * static_cast<uint64_t>(block_elems()));
    }

    BSRPageView<T> page_view(uint32_t page_id) {
        auto col_span = cols.page_span(page_id);
        auto val_span = values.page_span(page_id);
        const uint32_t blocks = std::min<uint32_t>(
            col_span.length,
            static_cast<uint32_t>(val_span.length / std::max<size_t>(block_elems(), 1)));
        return BSRPageView<T>{
            col_span.data,
            val_span.data,
            blocks,
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(block_elems()),
            page_id,
            col_span.global_begin};
    }

    BSRPageView<const T> page_view(uint32_t page_id) const {
        auto col_span = cols.page_span(page_id);
        auto val_span = values.page_span(page_id);
        const uint32_t blocks = std::min<uint32_t>(
            col_span.length,
            static_cast<uint32_t>(val_span.length / std::max<size_t>(block_elems(), 1)));
        return BSRPageView<const T>{
            col_span.data,
            val_span.data,
            blocks,
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(block_elems()),
            page_id,
            col_span.global_begin};
    }

    template <typename Fn>
    void for_each_page_view(Fn&& fn) {
        for (uint32_t page_id = 0; page_id < cols.page_count(); ++page_id) {
            fn(page_view(page_id));
        }
    }

    template <typename Fn>
    void for_each_page_view(Fn&& fn) const {
        for (uint32_t page_id = 0; page_id < cols.page_count(); ++page_id) {
            fn(page_view(page_id));
        }
    }

    template <typename Fn>
    void for_each_row_segment(const std::vector<int>& row_ptr, int row, Fn&& fn) const {
        uint64_t current = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]);
        const uint64_t end = static_cast<uint64_t>(row_ptr[static_cast<size_t>(row) + 1]);
        const uint32_t page_capacity = cols.page_capacity();
        while (current < end) {
            const uint32_t page_id = static_cast<uint32_t>(current / page_capacity);
            const uint32_t local_offset = static_cast<uint32_t>(current % page_capacity);
            const auto view = page_view(page_id);
            const uint32_t chunk = static_cast<uint32_t>(
                std::min<uint64_t>(view.nblocks - local_offset, end - current));
            fn(
                BSRPageView<const T>{
                    view.cols + local_offset,
                    view.vals + static_cast<size_t>(local_offset) * block_elems(),
                    chunk,
                    view.bsz,
                    view.block_elems,
                    view.page_id,
                    current},
                BSRRowSegment<T>{
                    row,
                    view.page_id,
                    current,
                    local_offset,
                    chunk,
                    current == static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]),
                    current + chunk == end});
            current += chunk;
        }
    }
};

template <typename T, typename Kernel>
struct VBCSRMatrixBackend {
    static constexpr size_t kDefaultPayloadPageElements = 1ULL << 22;
    static constexpr uint64_t kShapeBits = 16;
    static constexpr uint64_t kPageBits = 24;
    static constexpr uint64_t kSlotBits = 24;
    static constexpr uint64_t kShapeShift = kPageBits + kSlotBits;
    static constexpr uint64_t kPageShift = kSlotBits;
    static constexpr uint64_t kShapeMask = (uint64_t(1) << kShapeBits) - 1;
    static constexpr uint64_t kPageMask = (uint64_t(1) << kPageBits) - 1;
    static constexpr uint64_t kSlotMask = (uint64_t(1) << kSlotBits) - 1;

    enum class ExecutionKind {
        StaticFallback,
        BatchedFallback,
        JIT
    };

    struct ShapePage {
        int shape_id = -1;
        int page_id = -1;
        size_t slot_elems = 0;
        uint32_t slot_capacity = 0;
        uint32_t live_count = 0;
        uint32_t next_unused_slot = 0;
        std::unique_ptr<T[]> data;
        std::vector<uint32_t> free_slots;
        std::vector<uint32_t> live_slots;
        std::vector<uint32_t> slot_live_pos;
        std::vector<int> slot_logical_slots;
    };

    struct ShapeExecutionPolicy {
        std::atomic<uint64_t> apply_batches{0};
        std::atomic<uint64_t> apply_blocks{0};
        std::atomic<ExecutionKind> preferred_execution{ExecutionKind::StaticFallback};
    };

    struct SpMMExecutionPolicy {
        int row_dim = 0;
        int inner_dim = 0;
        int col_dim = 0;
        std::atomic<uint64_t> launched_batches{0};
        std::atomic<uint64_t> launched_products{0};
        std::atomic<ExecutionKind> preferred_execution{ExecutionKind::StaticFallback};
    };

    struct ShapeStorageEntry {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t slot_elems = 0;
        uint32_t page_slot_capacity = 0;
        size_t active_block_count = 0;
        size_t active_page_count = 0;
        std::vector<ShapePage> pages;
        std::vector<uint32_t> pages_with_free_slots;
        std::unique_ptr<ShapeExecutionPolicy> policy;
    };

    struct ShapeRegistryEntry {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t live_slots = 0;
        size_t live_elements = 0;
        std::vector<int> slots;
    };

    struct ShapeBatchView {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        int page_id = -1;
        const ShapePage* page = nullptr;
        const ShapeExecutionPolicy* policy = nullptr;

        uint32_t block_count() const {
            return page == nullptr ? 0u : static_cast<uint32_t>(page->live_slots.size());
        }

        int logical_slot(uint32_t block_index) const {
            const uint32_t page_slot = page->live_slots.at(block_index);
            return page->slot_logical_slots.at(page_slot);
        }

        const T* block_ptr(uint32_t block_index) const {
            const uint32_t page_slot = page->live_slots.at(block_index);
            return page->data.get() + static_cast<size_t>(page_slot) * page->slot_elems;
        }
    };

    struct SpMMPolicyKey {
        int row_dim = 0;
        int inner_dim = 0;
        int col_dim = 0;

        bool operator<(const SpMMPolicyKey& other) const {
            return std::tie(row_dim, inner_dim, col_dim) <
                   std::tie(other.row_dim, other.inner_dim, other.col_dim);
        }
    };

    VBCSRMatrixBackend() = default;
    VBCSRMatrixBackend(const VBCSRMatrixBackend&) = delete;
    VBCSRMatrixBackend& operator=(const VBCSRMatrixBackend&) = delete;

    VBCSRMatrixBackend(VBCSRMatrixBackend&& other) noexcept
        : blk_handles(std::move(other.blk_handles)),
          slot_shape_ids(std::move(other.slot_shape_ids)),
          shape_storage(std::move(other.shape_storage)),
          shape_registry(std::move(other.shape_registry)),
          shape_lookup(std::move(other.shape_lookup)),
          contiguous_layout(other.contiguous_layout),
          spmm_policy_lookup(std::move(other.spmm_policy_lookup)),
          spmm_policy_records(std::move(other.spmm_policy_records)) {}

    VBCSRMatrixBackend& operator=(VBCSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            blk_handles = std::move(other.blk_handles);
            slot_shape_ids = std::move(other.slot_shape_ids);
            shape_storage = std::move(other.shape_storage);
            shape_registry = std::move(other.shape_registry);
            shape_lookup = std::move(other.shape_lookup);
            contiguous_layout = other.contiguous_layout;
            std::lock_guard<std::mutex> lock(policy_mutex);
            spmm_policy_lookup = std::move(other.spmm_policy_lookup);
            spmm_policy_records = std::move(other.spmm_policy_records);
        }
        return *this;
    }

    // Per-logical-slot payload handle. The slot index is the flat local nonzero-block
    // index from row_ptr/col_ind; the handle locates that slot's payload page storage.
    std::vector<uint64_t> blk_handles;
    // Per-logical-slot shape-class id, aligned with blk_handles and logical slots.
    std::vector<int> slot_shape_ids;
    std::vector<ShapeStorageEntry> shape_storage;
    std::vector<ShapeRegistryEntry> shape_registry;
    std::map<std::pair<int, int>, int> shape_lookup;
    bool contiguous_layout = false;
    mutable std::mutex policy_mutex;
    mutable std::map<SpMMPolicyKey, size_t> spmm_policy_lookup;
    mutable std::vector<std::unique_ptr<SpMMExecutionPolicy>> spmm_policy_records;

    size_t local_scalar_nnz() const {
        size_t total = 0;
        for (const auto& entry : shape_registry) {
            total += entry.live_elements;
        }
        return total;
    }

    int shape_class_count() const {
        return static_cast<int>(shape_registry.size());
    }

    ExecutionKind execution_kind_for_shape(int shape_id) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shape_storage.size()) ||
            !shape_storage[shape_id].policy) {
            return ExecutionKind::StaticFallback;
        }
        return shape_storage[shape_id].policy->preferred_execution.load(std::memory_order_relaxed);
    }

    ExecutionKind execution_kind_for_spmm_triple(int row_dim, int inner_dim, int col_dim) const {
        const auto* policy = ensure_spmm_policy(row_dim, inner_dim, col_dim);
        if (policy == nullptr) {
            return ExecutionKind::StaticFallback;
        }
        return policy->preferred_execution.load(std::memory_order_relaxed);
    }

    void record_apply_batch(int shape_id, size_t block_count) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shape_storage.size()) ||
            !shape_storage[shape_id].policy) {
            return;
        }
        auto& entry = *shape_storage[shape_id].policy;
        entry.apply_batches.fetch_add(1, std::memory_order_relaxed);
        entry.apply_blocks.fetch_add(static_cast<uint64_t>(block_count), std::memory_order_relaxed);
    }

    void record_spmm_batch(int row_dim, int inner_dim, int col_dim, size_t product_count) const {
        auto* entry = ensure_spmm_policy(row_dim, inner_dim, col_dim);
        if (entry == nullptr) {
            return;
        }
        entry->launched_batches.fetch_add(1, std::memory_order_relaxed);
        entry->launched_products.fetch_add(static_cast<uint64_t>(product_count), std::memory_order_relaxed);
    }

    size_t shape_apply_batch_count(int shape_id) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shape_storage.size()) ||
            !shape_storage[shape_id].policy) {
            return 0;
        }
        return static_cast<size_t>(
            shape_storage[shape_id].policy->apply_batches.load(std::memory_order_relaxed));
    }

    size_t spmm_batch_count(int row_dim, int inner_dim, int col_dim) const {
        const auto* policy = ensure_spmm_policy(row_dim, inner_dim, col_dim);
        return policy == nullptr
            ? 0
            : static_cast<size_t>(policy->launched_batches.load(std::memory_order_relaxed));
    }

    bool is_contiguous() const {
        return contiguous_layout;
    }

    void mark_noncontiguous() {
        contiguous_layout = false;
    }

    template <typename GraphLike>
    void pack_contiguous(
        const GraphLike* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        rebuild_shape_registry(graph, row_ptr, col_ind);

        std::vector<int> registry_indices(shape_storage.size(), -1);
        for (int registry_idx = 0; registry_idx < static_cast<int>(shape_registry.size()); ++registry_idx) {
            registry_indices[shape_registry[registry_idx].shape_id] = registry_idx;
        }

        for (auto& storage_entry : shape_storage) {
            const ShapeRegistryEntry* registry_entry = nullptr;
            if (storage_entry.shape_id >= 0 &&
                storage_entry.shape_id < static_cast<int>(registry_indices.size()) &&
                registry_indices[storage_entry.shape_id] >= 0) {
                registry_entry = &shape_registry[registry_indices[storage_entry.shape_id]];
            }

            std::vector<ShapePage> packed_pages;
            packed_pages.reserve(storage_entry.active_page_count);

            if (registry_entry != nullptr && !registry_entry->slots.empty()) {
                for (int logical_slot : registry_entry->slots) {
                    const uint64_t old_handle = blk_handles.at(static_cast<size_t>(logical_slot));
                    const int old_page_id = decode_page_id(old_handle);
                    const uint32_t old_slot_id = decode_slot_id(old_handle);
                    const auto& old_page = storage_entry.pages.at(static_cast<size_t>(old_page_id));
                    const T* old_ptr =
                        old_page.data.get() + static_cast<size_t>(old_slot_id) * storage_entry.slot_elems;

                    if (packed_pages.empty() ||
                        packed_pages.back().next_unused_slot >= packed_pages.back().slot_capacity) {
                        packed_pages.push_back(make_shape_page(
                            storage_entry.shape_id,
                            static_cast<int>(packed_pages.size()),
                            storage_entry.slot_elems,
                            storage_entry.page_slot_capacity));
                    }

                    auto& new_page = packed_pages.back();
                    const uint32_t new_slot_id = new_page.next_unused_slot++;
                    T* new_ptr =
                        new_page.data.get() + static_cast<size_t>(new_slot_id) * storage_entry.slot_elems;
                    std::memcpy(new_ptr, old_ptr, storage_entry.slot_elems * sizeof(T));
                    new_page.slot_live_pos[new_slot_id] = static_cast<uint32_t>(new_page.live_slots.size());
                    new_page.live_slots.push_back(new_slot_id);
                    new_page.slot_logical_slots[new_slot_id] = logical_slot;
                    ++new_page.live_count;
                    blk_handles[static_cast<size_t>(logical_slot)] = encode_handle(
                        storage_entry.shape_id,
                        new_page.page_id,
                        new_slot_id);
                }
            }

            storage_entry.pages = std::move(packed_pages);
            storage_entry.pages_with_free_slots.clear();
            if (!storage_entry.pages.empty() &&
                storage_entry.pages.back().live_count < storage_entry.pages.back().slot_capacity) {
                storage_entry.pages_with_free_slots.push_back(
                    static_cast<uint32_t>(storage_entry.pages.back().page_id));
            }

            storage_entry.active_block_count =
                registry_entry == nullptr ? 0 : static_cast<size_t>(registry_entry->live_slots);
            storage_entry.active_page_count = storage_entry.pages.size();
        }

        rebuild_shape_registry(graph, row_ptr, col_ind);
        contiguous_layout = true;
    }

    template <typename Fn>
    void for_each_shape_class(Fn&& fn) const {
        for (const auto& entry : shape_registry) {
            fn(entry.shape_id, entry.row_dim, entry.col_dim, entry.slots);
        }
    }

    template <typename Fn>
    void for_each_shape_page(Fn&& fn) const {
        for (const auto& entry : shape_storage) {
            for (const auto& page : entry.pages) {
                if (page.live_count == 0) {
                    continue;
                }
                fn(entry.shape_id, entry.row_dim, entry.col_dim, page);
            }
        }
    }

    template <typename Fn>
    void for_each_shape_batch(Fn&& fn) const {
        for (const auto& entry : shape_storage) {
            if (entry.shape_id < 0 || !entry.policy) {
                continue;
            }
            for (const auto& page : entry.pages) {
                if (page.live_count == 0 || page.live_slots.empty()) {
                    continue;
                }
                ShapeBatchView view{
                    entry.shape_id,
                    entry.row_dim,
                    entry.col_dim,
                    page.page_id,
                    &page,
                    entry.policy.get()};
                if constexpr (std::is_invocable_v<Fn, const ShapeBatchView&>) {
                    fn(view);
                } else {
                    std::vector<int> logical_slots;
                    logical_slots.reserve(page.live_slots.size());
                    for (uint32_t live_slot : page.live_slots) {
                        logical_slots.push_back(page.slot_logical_slots.at(live_slot));
                    }
                    fn(entry.shape_id, entry.row_dim, entry.col_dim, page.page_id, logical_slots);
                }
            }
        }
    }

    static uint64_t encode_handle(int shape_id, int page_id, uint32_t slot_id) {
        if (shape_id < 0 || page_id < 0) {
            throw std::logic_error("VBCSR handle cannot encode negative indices");
        }
        if (static_cast<uint64_t>(shape_id) > kShapeMask ||
            static_cast<uint64_t>(page_id) > kPageMask ||
            static_cast<uint64_t>(slot_id) > kSlotMask) {
            throw std::overflow_error("VBCSR handle field overflow");
        }
        return (static_cast<uint64_t>(shape_id) << kShapeShift) |
               (static_cast<uint64_t>(page_id) << kPageShift) |
               static_cast<uint64_t>(slot_id);
    }

    static int decode_shape_id(uint64_t handle) {
        return static_cast<int>((handle >> kShapeShift) & kShapeMask);
    }

    static int decode_page_id(uint64_t handle) {
        return static_cast<int>((handle >> kPageShift) & kPageMask);
    }

    static uint32_t decode_slot_id(uint64_t handle) {
        return static_cast<uint32_t>(handle & kSlotMask);
    }

    int ensure_shape(int row_dim, int col_dim) {
        const auto key = std::make_pair(row_dim, col_dim);
        auto it = shape_lookup.find(key);
        if (it != shape_lookup.end()) {
            return it->second;
        }

        const int shape_id = static_cast<int>(shape_storage.size());
        if (static_cast<uint64_t>(shape_id) > kShapeMask) {
            throw std::overflow_error("VBCSR shape registry exceeded handle capacity");
        }

        ShapeStorageEntry entry;
        entry.shape_id = shape_id;
        entry.row_dim = row_dim;
        entry.col_dim = col_dim;
        entry.slot_elems = static_cast<size_t>(row_dim) * static_cast<size_t>(col_dim);
        entry.page_slot_capacity = default_page_slot_capacity(entry.slot_elems);
        entry.policy = std::make_unique<ShapeExecutionPolicy>();
        shape_storage.push_back(std::move(entry));
        shape_lookup.emplace(key, shape_id);
        return shape_id;
    }

    uint64_t allocate_slot(int row_dim, int col_dim) {
        return allocate_slot_for_shape(ensure_shape(row_dim, col_dim));
    }

    void bind_logical_slot(int logical_slot, int shape_id, uint64_t handle) {
        contiguous_layout = false;
        if (logical_slot < 0) {
            throw std::logic_error("VBCSR logical slot cannot be negative");
        }
        if (static_cast<size_t>(logical_slot) >= blk_handles.size()) {
            blk_handles.resize(static_cast<size_t>(logical_slot) + 1, 0);
        }
        if (static_cast<size_t>(logical_slot) >= slot_shape_ids.size()) {
            slot_shape_ids.resize(static_cast<size_t>(logical_slot) + 1, -1);
        }
        blk_handles[logical_slot] = handle;
        slot_shape_ids[logical_slot] = shape_id;

        const int page_id = decode_page_id(handle);
        const uint32_t slot_id = decode_slot_id(handle);
        auto& page = shape_storage.at(shape_id).pages.at(page_id);
        page.slot_logical_slots.at(slot_id) = logical_slot;
    }

    uint64_t allocate_slot_for_shape(int shape_id) {
        contiguous_layout = false;
        auto& entry = shape_storage.at(shape_id);
        while (!entry.pages_with_free_slots.empty()) {
            const uint32_t candidate = entry.pages_with_free_slots.back();
            if (candidate < entry.pages.size() && page_has_free(entry.pages[candidate])) {
                break;
            }
            entry.pages_with_free_slots.pop_back();
        }

        if (entry.pages_with_free_slots.empty()) {
            const uint32_t new_page_id = create_page(entry);
            entry.pages_with_free_slots.push_back(new_page_id);
        }

        const uint32_t page_id = entry.pages_with_free_slots.back();
        auto& page = entry.pages[page_id];
        const uint32_t slot_id = allocate_from_page(page);
        ++entry.active_block_count;
        if (page.live_count == 1) {
            ++entry.active_page_count;
        }
        if (!page_has_free(page)) {
            entry.pages_with_free_slots.pop_back();
        }
        return encode_handle(shape_id, static_cast<int>(page_id), slot_id);
    }

    void free_handle(uint64_t handle) {
        contiguous_layout = false;
        const int shape_id = decode_shape_id(handle);
        const int page_id = decode_page_id(handle);
        const uint32_t slot_id = decode_slot_id(handle);

        auto& entry = shape_storage.at(shape_id);
        auto& page = entry.pages.at(page_id);
        const bool was_full = !page_has_free(page);

        T* ptr = page.data.get() + static_cast<size_t>(slot_id) * entry.slot_elems;
        std::fill(ptr, ptr + entry.slot_elems, T(0));

        const uint32_t live_pos = page.slot_live_pos.at(slot_id);
        if (live_pos == std::numeric_limits<uint32_t>::max()) {
            throw std::logic_error("Attempted to free an inactive VBCSR slot");
        }

        const uint32_t back_slot = page.live_slots.back();
        page.live_slots[live_pos] = back_slot;
        page.slot_live_pos[back_slot] = live_pos;
        page.live_slots.pop_back();
        page.slot_live_pos[slot_id] = std::numeric_limits<uint32_t>::max();
        page.slot_logical_slots[slot_id] = -1;
        page.free_slots.push_back(slot_id);
        --page.live_count;
        --entry.active_block_count;
        if (page.live_count == 0) {
            --entry.active_page_count;
        }
        if (was_full) {
            entry.pages_with_free_slots.push_back(static_cast<uint32_t>(page_id));
        }
    }

    T* get_ptr(uint64_t handle) {
        const auto& self = *this;
        return const_cast<T*>(self.get_ptr(handle));
    }

    const T* get_ptr(uint64_t handle) const {
        const int shape_id = decode_shape_id(handle);
        const int page_id = decode_page_id(handle);
        const uint32_t slot_id = decode_slot_id(handle);
        const auto& entry = shape_storage.at(shape_id);
        const auto& page = entry.pages.at(page_id);
        return page.data.get() + static_cast<size_t>(slot_id) * entry.slot_elems;
    }

    size_t block_size_elements(int slot) const {
        return slot_elems_for_shape(slot_shape_ids.at(slot));
    }

    template <typename GraphLike>
    void rebuild_shape_registry(
        const GraphLike* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        slot_shape_ids.assign(col_ind.size(), -1);
        shape_registry.clear();
        if (graph == nullptr) {
            return;
        }

        for (auto& storage_entry : shape_storage) {
            for (auto& page : storage_entry.pages) {
                std::fill(page.slot_logical_slots.begin(), page.slot_logical_slots.end(), -1);
            }
        }

        std::vector<int> registry_indices(shape_storage.size(), -1);
        const int n_rows = static_cast<int>(row_ptr.size()) - 1;
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph->block_sizes[row];
            for (int slot = row_ptr[row]; slot < row_ptr[row + 1]; ++slot) {
                const int col = col_ind[slot];
                const int col_dim = graph->block_sizes[col];
                const int shape_id = ensure_shape(row_dim, col_dim);
                if (shape_id >= static_cast<int>(registry_indices.size())) {
                    registry_indices.resize(shape_storage.size(), -1);
                }

                if (registry_indices[shape_id] < 0) {
                    const auto& storage_entry = shape_storage[shape_id];
                    registry_indices[shape_id] = static_cast<int>(shape_registry.size());
                    shape_registry.push_back(ShapeRegistryEntry{
                        shape_id,
                        storage_entry.row_dim,
                        storage_entry.col_dim,
                        0,
                        0,
                        {}});
                }

                slot_shape_ids[slot] = shape_id;
                const uint64_t handle = blk_handles.at(slot);
                const int page_id = decode_page_id(handle);
                const uint32_t page_slot = decode_slot_id(handle);
                shape_storage[shape_id].pages.at(page_id).slot_logical_slots.at(page_slot) = slot;
                auto& entry = shape_registry[registry_indices[shape_id]];
                entry.slots.push_back(slot);
                ++entry.live_slots;
                entry.live_elements += slot_elems_for_shape(shape_id);
            }
        }
    }

private:
    static ShapePage make_shape_page(int shape_id, int page_id, size_t slot_elems, uint32_t slot_capacity) {
        ShapePage page;
        page.shape_id = shape_id;
        page.page_id = page_id;
        page.slot_elems = slot_elems;
        page.slot_capacity = slot_capacity;
        page.live_count = 0;
        page.next_unused_slot = 0;
        page.data = std::make_unique<T[]>(slot_elems * static_cast<size_t>(slot_capacity));
        std::fill(page.data.get(), page.data.get() + slot_elems * static_cast<size_t>(slot_capacity), T(0));
        page.slot_live_pos.assign(slot_capacity, std::numeric_limits<uint32_t>::max());
        page.slot_logical_slots.assign(slot_capacity, -1);
        return page;
    }

    static bool page_has_free(const ShapePage& page) {
        return !page.free_slots.empty() || page.next_unused_slot < page.slot_capacity;
    }

    static uint32_t default_page_slot_capacity(size_t slot_elems) {
        if (slot_elems == 0) {
            return 1;
        }
        const size_t page_elems = std::max<size_t>(1, kDefaultPayloadPageElements / slot_elems);
        const size_t bounded = std::min<size_t>(page_elems, static_cast<size_t>(kSlotMask));
        return static_cast<uint32_t>(std::max<size_t>(bounded, 1));
    }

    static uint32_t create_page(ShapeStorageEntry& entry) {
        entry.pages.push_back(make_shape_page(
            entry.shape_id,
            static_cast<int>(entry.pages.size()),
            entry.slot_elems,
            entry.page_slot_capacity));
        return static_cast<uint32_t>(entry.pages.size() - 1);
    }

    static uint32_t allocate_from_page(ShapePage& page) {
        uint32_t slot_id = 0;
        if (!page.free_slots.empty()) {
            slot_id = page.free_slots.back();
            page.free_slots.pop_back();
        } else {
            if (page.next_unused_slot >= page.slot_capacity) {
                throw std::overflow_error("VBCSR shape page exhausted without free slot");
            }
            slot_id = page.next_unused_slot++;
        }
        page.slot_live_pos[slot_id] = static_cast<uint32_t>(page.live_slots.size());
        page.live_slots.push_back(slot_id);
        ++page.live_count;
        return slot_id;
    }

    size_t slot_elems_for_shape(int shape_id) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shape_storage.size())) {
            throw std::logic_error("Invalid VBCSR shape identifier");
        }
        return shape_storage[shape_id].slot_elems;
    }

    SpMMExecutionPolicy* ensure_spmm_policy(int row_dim, int inner_dim, int col_dim) const {
        const SpMMPolicyKey key{row_dim, inner_dim, col_dim};
        std::lock_guard<std::mutex> lock(policy_mutex);
        auto it = spmm_policy_lookup.find(key);
        if (it != spmm_policy_lookup.end()) {
            return spmm_policy_records[it->second].get();
        }

        auto record = std::make_unique<SpMMExecutionPolicy>();
        record->row_dim = row_dim;
        record->inner_dim = inner_dim;
        record->col_dim = col_dim;
        const size_t record_index = spmm_policy_records.size();
        spmm_policy_records.push_back(std::move(record));
        spmm_policy_lookup.emplace(key, record_index);
        return spmm_policy_records.back().get();
    }
};

template <typename T, typename Kernel>
using MatrixBackendHandle = std::variant<
    std::monostate,
    CSRMatrixBackend<T>,
    BSRMatrixBackend<T>,
    VBCSRMatrixBackend<T, Kernel>>;

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_csr_backend_handle(CSRMatrixBackend<T> storage) {
    return MatrixBackendHandle<T, Kernel>(std::in_place_type<CSRMatrixBackend<T>>, std::move(storage));
}

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_bsr_backend_handle(BSRMatrixBackend<T> storage) {
    return MatrixBackendHandle<T, Kernel>(std::in_place_type<BSRMatrixBackend<T>>, std::move(storage));
}

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_vbcsr_backend_handle(VBCSRMatrixBackend<T, Kernel> storage) {
    return MatrixBackendHandle<T, Kernel>(std::in_place_type<VBCSRMatrixBackend<T, Kernel>>, std::move(storage));
}

template <typename T, typename Kernel>
CSRMatrixBackend<T>& require_csr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<CSRMatrixBackend<T>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the CSR storage path");
    }
    return *storage;
}

template <typename T, typename Kernel>
const CSRMatrixBackend<T>& require_csr_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<CSRMatrixBackend<T>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the CSR storage path");
    }
    return *storage;
}

template <typename T, typename Kernel>
BSRMatrixBackend<T>& require_bsr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<BSRMatrixBackend<T>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the BSR storage path");
    }
    return *storage;
}

template <typename T, typename Kernel>
const BSRMatrixBackend<T>& require_bsr_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<BSRMatrixBackend<T>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the BSR storage path");
    }
    return *storage;
}

template <typename T, typename Kernel>
VBCSRMatrixBackend<T, Kernel>& require_vbcsr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<VBCSRMatrixBackend<T, Kernel>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the VBCSR storage path");
    }
    return *storage;
}

template <typename T, typename Kernel>
const VBCSRMatrixBackend<T, Kernel>& require_vbcsr_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* storage = std::get_if<VBCSRMatrixBackend<T, Kernel>>(&handle);
    if (storage == nullptr) {
        throw std::logic_error("Active backend is not the VBCSR storage path");
    }
    return *storage;
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_HANDLE_HPP
