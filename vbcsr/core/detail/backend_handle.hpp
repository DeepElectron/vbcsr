#ifndef VBCSR_DETAIL_BACKEND_HANDLE_HPP
#define VBCSR_DETAIL_BACKEND_HANDLE_HPP

#include "paged_array.hpp"
#include "shape_paged_storage.hpp"

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
struct CSRPageSlice {
    const int* cols = nullptr;
    T* values = nullptr;
    uint32_t nnz_count = 0;
    uint32_t page_index = 0;
    uint64_t first_nnz = 0;
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
    uint32_t page_index = 0;
    uint64_t first_nnz = 0;
    uint32_t nnz_count = 0;
    // Inclusive first local row touched by this page's [first_nnz, first_nnz + nnz_count) interval.
    int row_begin = 0;
    // Exclusive one-past-last local row touched by this page.
    int row_end = 0;
};

#ifdef VBCSR_HAVE_MKL_SPARSE
struct CSRVendorMKLMMVariant {
    int num_rhs = -1;
    uint64_t last_use = 0;
    sparse_matrix_t handle = nullptr;

    CSRVendorMKLMMVariant() = default;
    CSRVendorMKLMMVariant(const CSRVendorMKLMMVariant&) = delete;
    CSRVendorMKLMMVariant& operator=(const CSRVendorMKLMMVariant&) = delete;

    CSRVendorMKLMMVariant(CSRVendorMKLMMVariant&& other) noexcept
        : num_rhs(other.num_rhs),
          last_use(other.last_use),
          handle(other.handle) {
        other.num_rhs = -1;
        other.last_use = 0;
        other.handle = nullptr;
    }

    CSRVendorMKLMMVariant& operator=(CSRVendorMKLMMVariant&& other) noexcept {
        if (this != &other) {
            reset();
            num_rhs = other.num_rhs;
            last_use = other.last_use;
            handle = other.handle;
            other.num_rhs = -1;
            other.last_use = 0;
            other.handle = nullptr;
        }
        return *this;
    }

    ~CSRVendorMKLMMVariant() {
        reset();
    }

    void reset() {
        if (handle != nullptr) {
            mkl_sparse_destroy(handle);
            handle = nullptr;
        }
        num_rhs = -1;
        last_use = 0;
    }
};

struct CSRVendorMKLHandle {
    static constexpr size_t kMaxMMVariants = 4;
    static constexpr size_t kInvalidMMVariantIndex = static_cast<size_t>(-1);

    sparse_matrix_t mv_handle = nullptr;
    std::vector<CSRVendorMKLMMVariant> mm_variants;
    uint64_t use_clock = 0;
    size_t recent_mm_variant_index = kInvalidMMVariantIndex;

    CSRVendorMKLHandle() = default;
    CSRVendorMKLHandle(const CSRVendorMKLHandle&) = delete;
    CSRVendorMKLHandle& operator=(const CSRVendorMKLHandle&) = delete;

    CSRVendorMKLHandle(CSRVendorMKLHandle&& other) noexcept
        : mv_handle(other.mv_handle),
          mm_variants(std::move(other.mm_variants)),
          use_clock(other.use_clock),
          recent_mm_variant_index(other.recent_mm_variant_index) {
        other.mv_handle = nullptr;
        other.use_clock = 0;
        other.recent_mm_variant_index = kInvalidMMVariantIndex;
    }

    CSRVendorMKLHandle& operator=(CSRVendorMKLHandle&& other) noexcept {
        if (this != &other) {
            reset();
            mv_handle = other.mv_handle;
            mm_variants = std::move(other.mm_variants);
            use_clock = other.use_clock;
            recent_mm_variant_index = other.recent_mm_variant_index;
            other.mv_handle = nullptr;
            other.use_clock = 0;
            other.recent_mm_variant_index = kInvalidMMVariantIndex;
        }
        return *this;
    }

    ~CSRVendorMKLHandle() {
        reset();
    }

    void reset() {
        if (mv_handle != nullptr) {
            mkl_sparse_destroy(mv_handle);
            mv_handle = nullptr;
        }
        mm_variants.clear();
        use_clock = 0;
        recent_mm_variant_index = kInvalidMMVariantIndex;
    }

    CSRVendorMKLMMVariant* find_mm_variant(int num_rhs) {
        if (recent_mm_variant_index < mm_variants.size() &&
            mm_variants[recent_mm_variant_index].num_rhs == num_rhs) {
            return &mm_variants[recent_mm_variant_index];
        }
        for (auto& variant : mm_variants) {
            if (variant.num_rhs == num_rhs) {
                recent_mm_variant_index = static_cast<size_t>(&variant - mm_variants.data());
                return &variant;
            }
        }
        return nullptr;
    }

    const CSRVendorMKLMMVariant* find_mm_variant(int num_rhs) const {
        if (recent_mm_variant_index < mm_variants.size() &&
            mm_variants[recent_mm_variant_index].num_rhs == num_rhs) {
            return &mm_variants[recent_mm_variant_index];
        }
        for (size_t idx = 0; idx < mm_variants.size(); ++idx) {
            if (mm_variants[idx].num_rhs == num_rhs) {
                return &mm_variants[idx];
            }
        }
        return nullptr;
    }

    sparse_matrix_t mm_handle(int num_rhs) const {
        const auto* variant = find_mm_variant(num_rhs);
        return variant == nullptr ? nullptr : variant->handle;
    }

    uint64_t next_use_stamp() {
        return ++use_clock;
    }

    void remember_mm_variant(size_t variant_index) {
        recent_mm_variant_index = variant_index;
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
    CSRVendorPageMetadata metadata;
    // Page-local CSR row pointer with row indices rebased to metadata.row_begin and
    // NNZ offsets rebased to metadata.first_nnz.
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
    static constexpr uint32_t page_size_limit() {
        constexpr uint64_t hard_limit =
            std::min<uint64_t>(
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                static_cast<uint64_t>(std::numeric_limits<int>::max()));
        return static_cast<uint32_t>(hard_limit);
    }

    static uint32_t clamp_page_size(uint32_t requested) {
        if (requested == 0) {
            return page_size_limit();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(requested, 1u, static_cast<uint64_t>(page_size_limit())));
    }

    PagedBuffer<T> values;
    mutable std::unique_ptr<CSRVendorCache<T>> vendor_cache; // one with the metadata and handler
    mutable std::mutex vendor_cache_mutex;
    mutable std::atomic<uint64_t> vendor_launch_count{0};
    uint32_t page_setting_ = page_size_limit();

    CSRMatrixBackend()
        : values(page_size_limit()) {}

    explicit CSRMatrixBackend(uint32_t page_size)
        : values(clamp_page_size(page_size)),
          page_setting_(clamp_page_size(page_size)) {}

    CSRMatrixBackend(const CSRMatrixBackend&) = delete;
    CSRMatrixBackend& operator=(const CSRMatrixBackend&) = delete;

    CSRMatrixBackend(CSRMatrixBackend&& other) noexcept
        : values(std::move(other.values)),
          page_setting_(other.page_setting_) {
        other.vendor_launch_count.store(0, std::memory_order_release);
        other.page_setting_ = page_size_limit();
    }

    CSRMatrixBackend& operator=(CSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            values = std::move(other.values);
            page_setting_ = other.page_setting_;
            invalidate_vendor_cache();
            vendor_launch_count.store(0, std::memory_order_release);
            other.vendor_launch_count.store(0, std::memory_order_release);
            other.page_setting_ = page_size_limit();
        }
        return *this;
    }

    uint32_t page_setting() const {
        return page_setting_;
    }

    void set_page_setting(uint32_t page_size) {
        page_setting_ = clamp_page_size(page_size);
    }

    uint32_t page_size() const {
        return values.elements_per_page();
    }

    uint32_t page_count() const {
        return values.page_count();
    }

    uint64_t nnz_count() const {
        return values.size();
    }

    void initialize_structure(uint64_t logical_nnz) {
        rebuild_pages(logical_nnz);
        values.resize(logical_nnz);
    }

    void initialize_structure(uint64_t logical_nnz, uint32_t page_size) {
        set_page_setting(page_size);
        initialize_structure(logical_nnz);
    }

    void initialize_structure(const std::vector<int>& col_ind) {
        initialize_structure(static_cast<uint64_t>(col_ind.size()));
    }

    void initialize_structure(const std::vector<int>& col_ind, uint32_t page_size) {
        initialize_structure(static_cast<uint64_t>(col_ind.size()), page_size);
    }

    T* value_ptr(int slot) {
        return values.element_ptr(static_cast<uint64_t>(slot));
    }

    const T* value_ptr(int slot) const {
        return values.element_ptr(static_cast<uint64_t>(slot));
    }

    CSRPageSlice<T> page(const std::vector<int>& col_ind, uint32_t page_index) {
        auto value_page = values.page(page_index);
        if (value_page.first_element + value_page.count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("CSRMatrixBackend::page column span out of bounds");
        }
        return CSRPageSlice<T>{
            col_ind.data() + value_page.first_element,
            value_page.data,
            value_page.count,
            page_index,
            value_page.first_element};
    }

    CSRPageSlice<const T> page(const std::vector<int>& col_ind, uint32_t page_index) const {
        auto value_page = values.page(page_index);
        if (value_page.first_element + value_page.count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("CSRMatrixBackend::page column span out of bounds");
        }
        return CSRPageSlice<const T>{
            col_ind.data() + value_page.first_element,
            value_page.data,
            value_page.count,
            page_index,
            value_page.first_element};
    }

    template <typename Fn>
    void for_each_page(const std::vector<int>& col_ind, Fn&& fn) {
        for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
            fn(page(col_ind, page_index));
        }
    }

    template <typename Fn>
    void for_each_page(const std::vector<int>& col_ind, Fn&& fn) const {
        for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
            fn(page(col_ind, page_index));
        }
    }

    template <typename Fn>
    void for_each_row_slice(const std::vector<int>& row_ptr, const std::vector<int>& col_ind, int row, Fn&& fn) const {
        auto [current, end] = row_nnz_bounds(row_ptr, row);
        const uint32_t page_capacity = values.elements_per_page();
        while (current < end) {
            const uint32_t page_index = static_cast<uint32_t>(current / page_capacity);
            const uint32_t local_offset = static_cast<uint32_t>(current % page_capacity);
            const auto page_slice = page(col_ind, page_index);
            const uint32_t nnz_count = static_cast<uint32_t>(
                std::min<uint64_t>(page_slice.nnz_count - local_offset, end - current));
            fn(trim_page_slice(page_slice, local_offset, nnz_count));
            current += nnz_count;
        }
    }

private:
    uint32_t active_page_size(uint64_t logical_nnz) const {
        if (logical_nnz == 0) {
            return page_setting_;
        }
        return static_cast<uint32_t>(std::min<uint64_t>(logical_nnz, page_setting_));
    }

    void rebuild_pages(uint64_t logical_nnz) {
        values = PagedBuffer<T>(active_page_size(logical_nnz));
        invalidate_vendor_cache();
    }

    static std::pair<uint64_t, uint64_t> row_nnz_bounds(const std::vector<int>& row_ptr, int row) {
        return {
            static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]),
            static_cast<uint64_t>(row_ptr[static_cast<size_t>(row) + 1])};
    }

    template <typename ValueType>
    static CSRPageSlice<ValueType> trim_page_slice(
        CSRPageSlice<ValueType> page_slice,
        uint32_t local_offset,
        uint32_t nnz_count) {
        if (local_offset > page_slice.nnz_count || nnz_count > page_slice.nnz_count - local_offset) {
            throw std::out_of_range("CSRMatrixBackend::trim_page_slice range out of bounds");
        }
        return CSRPageSlice<ValueType>{
            page_slice.cols + local_offset,
            page_slice.values + local_offset,
            nnz_count,
            page_slice.page_index,
            page_slice.first_nnz + local_offset};
    }

public:
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

    void note_vendor_launch(uint64_t page_calls = 1) const {
        vendor_launch_count.fetch_add(page_calls, std::memory_order_acq_rel);
    }

#ifdef VBCSR_HAVE_MKL_SPARSE
    bool ensure_mkl_mm_handles(const CSRVendorCache<T>& cache, const std::vector<int>& col_ind, int num_rhs) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (cache.kind != CSRVendorBackendKind::MKL || vendor_cache == nullptr || vendor_cache.get() != &cache) {
            return false;
        }

        for (auto& entry : vendor_cache->pages) {
            if (!ensure_mkl_mm_variant_locked(entry, col_ind, vendor_cache->num_cols, num_rhs)) {
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
        CSRPageSlice<const T> page_slice) {
        CSRVendorPageEntry<T> entry;
        entry.metadata.page_index = page_slice.page_index;
        entry.metadata.first_nnz = page_slice.first_nnz;
        entry.metadata.nnz_count = page_slice.nnz_count;
        if (page_slice.nnz_count == 0) {
            return entry;
        }

        const int begin = static_cast<int>(page_slice.first_nnz);
        const int end = begin + static_cast<int>(page_slice.nnz_count);
        // A storage page can begin or end in the middle of a CSR row. Find the first row
        // that owns NNZ >= begin and the first row whose start is >= end.
        const auto row_begin_it =
            std::upper_bound(row_ptr.begin() + 1, row_ptr.end(), begin);
        const auto row_end_it =
            std::lower_bound(row_ptr.begin(), row_ptr.end() - 1, end);

        entry.metadata.row_begin = static_cast<int>(std::distance(row_ptr.begin(), row_begin_it)) - 1;
        entry.metadata.row_end = static_cast<int>(std::distance(row_ptr.begin(), row_end_it));

        entry.row_ptr.reserve(static_cast<size_t>(entry.metadata.row_end - entry.metadata.row_begin + 1));
        for (int row = entry.metadata.row_begin; row <= entry.metadata.row_end; ++row) {
            // Clip the global row_ptr entry to this page's NNZ interval, then rebase it so
            // the vendor handle sees a compact CSR matrix starting at NNZ offset 0.
            const int clamped = std::clamp(row_ptr[static_cast<size_t>(row)], begin, end);
            entry.row_ptr.push_back(clamped - begin);
        }
        return entry;
    }

#ifdef VBCSR_HAVE_MKL_SPARSE
    static void destroy_mkl_sparse_handle(sparse_matrix_t& handle) {
        if (handle != nullptr) {
            mkl_sparse_destroy(handle);
            handle = nullptr;
        }
    }

    static matrix_descr make_mkl_descr() {
        matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.mode = SPARSE_FILL_MODE_FULL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        return descr;
    }

    static sparse_operation_t mkl_adjoint_operation() {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        }
        return SPARSE_OPERATION_TRANSPOSE;
    }

    bool build_mkl_raw_handle(
        sparse_matrix_t& out_handle,
        const CSRVendorPageEntry<T>& entry,
        CSRPageSlice<const T> page_slice,
        int num_cols) const {
        destroy_mkl_sparse_handle(out_handle);

        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        const MKL_INT rows = static_cast<MKL_INT>(entry.row_count());
        const MKL_INT cols_count = static_cast<MKL_INT>(num_cols);
        auto* row_begin = reinterpret_cast<MKL_INT*>(const_cast<int*>(entry.row_ptr.data()));
        // MKL expects CSR row_begin/row_end arrays of length rows. Our compact page-local
        // row_ptr has length rows + 1, so row_end is just row_begin shifted by one entry.
        auto* row_end = row_begin + 1;
        auto* col_idx = reinterpret_cast<MKL_INT*>(const_cast<int*>(page_slice.cols));

        if constexpr (std::is_same_v<T, double>) {
            status = mkl_sparse_d_create_csr(
                &out_handle,
                SPARSE_INDEX_BASE_ZERO,
                rows,
                cols_count,
                row_begin,
                row_end,
                col_idx,
                const_cast<double*>(page_slice.values));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = mkl_sparse_z_create_csr(
                &out_handle,
                SPARSE_INDEX_BASE_ZERO,
                rows,
                cols_count,
                row_begin,
                row_end,
                col_idx,
                reinterpret_cast<MKL_Complex16*>(const_cast<std::complex<double>*>(page_slice.values)));
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
        const CSRVendorPageEntry<T>& entry,
        CSRPageSlice<const T> page_slice,
        int num_cols) const {
        sparse_matrix_t raw_handle = nullptr;
        if (!build_mkl_raw_handle(raw_handle, entry, page_slice, num_cols)) {
            return false;
        }

        const matrix_descr descr = make_mkl_descr();
        if (mkl_sparse_set_mv_hint(raw_handle, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1) !=
            SPARSE_STATUS_SUCCESS) {
            destroy_mkl_sparse_handle(raw_handle);
            return false;
        }
        if (mkl_sparse_set_mv_hint(raw_handle, mkl_adjoint_operation(), descr, 1) != SPARSE_STATUS_SUCCESS) {
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
        const CSRVendorPageEntry<T>& entry,
        CSRPageSlice<const T> page_slice,
        int num_cols,
        int num_rhs) const {
        sparse_matrix_t raw_handle = nullptr;
        if (!build_mkl_raw_handle(raw_handle, entry, page_slice, num_cols)) {
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
                mkl_adjoint_operation(),
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
        const std::vector<int>& col_ind,
        int num_cols,
        int num_rhs) const {
        if (auto* existing = entry.mkl.find_mm_variant(num_rhs)) {
            existing->last_use = entry.mkl.next_use_stamp();
            return true;
        }

        CSRVendorMKLMMVariant variant;
        if (!build_mkl_mm_variant(variant, entry, page(col_ind, entry.metadata.page_index), num_cols, num_rhs)) {
            return false;
        }
        variant.last_use = entry.mkl.next_use_stamp();

        if (entry.mkl.mm_variants.size() < CSRVendorMKLHandle::kMaxMMVariants) {
            entry.mkl.mm_variants.push_back(std::move(variant));
            entry.mkl.remember_mm_variant(entry.mkl.mm_variants.size() - 1);
            return true;
        }

        size_t lru_index = 0;
        for (size_t idx = 1; idx < entry.mkl.mm_variants.size(); ++idx) {
            if (entry.mkl.mm_variants[idx].last_use < entry.mkl.mm_variants[lru_index].last_use) {
                lru_index = idx;
            }
        }
        entry.mkl.mm_variants[lru_index] = std::move(variant);
        entry.mkl.remember_mm_variant(lru_index);
        return true;
    }

    bool build_mkl_page_handle(
        CSRVendorPageEntry<T>& entry,
        CSRPageSlice<const T> page_slice,
        int num_cols) const {
        return build_mkl_mv_handle(entry.mkl, entry, page_slice, num_cols);
    }
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
    bool build_aocl_page_handle(
        CSRVendorPageEntry<T>& entry,
        CSRPageSlice<const T> page_slice,
        aoclsparse_mat_descr descr,
        int num_cols) const {
        aoclsparse_status status = aoclsparse_status_not_implemented;
        const aoclsparse_int rows = static_cast<aoclsparse_int>(entry.row_count());
        const aoclsparse_int cols_count = static_cast<aoclsparse_int>(num_cols);
        const aoclsparse_int nnz = static_cast<aoclsparse_int>(page_slice.nnz_count);
        auto* row_ptr_local = reinterpret_cast<aoclsparse_int*>(entry.row_ptr.data());
        auto* col_idx = reinterpret_cast<aoclsparse_int*>(const_cast<int*>(page_slice.cols));

        if constexpr (std::is_same_v<T, double>) {
            status = aoclsparse_create_dcsr(
                &entry.aocl.handle,
                aoclsparse_index_base_zero,
                rows,
                cols_count,
                nnz,
                row_ptr_local,
                col_idx,
                const_cast<double*>(page_slice.values));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = aoclsparse_create_zcsr(
                &entry.aocl.handle,
                aoclsparse_index_base_zero,
                rows,
                cols_count,
                nnz,
                row_ptr_local,
                col_idx,
                reinterpret_cast<aoclsparse_double_complex*>(const_cast<std::complex<double>*>(page_slice.values)));
        } else {
            return false;
        }

        if (status != aoclsparse_status_success) {
            entry.aocl.reset();
            return false;
        }

        if constexpr (std::is_same_v<T, double>) {
            aoclsparse_set_mv_hint(entry.aocl.handle, aoclsparse_operation_none, descr, 1);
            aoclsparse_set_mv_hint(entry.aocl.handle, aoclsparse_operation_transpose, descr, 1);
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
            const auto page_slice = page(col_ind, entry.metadata.page_index);
            bool ok = false;
            switch (cache.kind) {
            case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
                ok = build_mkl_page_handle(entry, page_slice, num_cols);
#endif
                break;
            case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
                ok = build_aocl_page_handle(entry, page_slice, cache.aocl_descr.handle, num_cols);
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
struct BSRPageSlice {
    const int* cols = nullptr;
    T* values = nullptr;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t block_value_count = 0;
    uint32_t page_index = 0;
    uint64_t first_block = 0;
};

template <typename T>
struct BSRMatrixBackend {
    static uint32_t page_size_limit(int uniform_block_size) {
        if (uniform_block_size <= 0) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint64_t block_elems =
            static_cast<uint64_t>(uniform_block_size) * static_cast<uint64_t>(uniform_block_size);
        const uint64_t by_values =
            block_elems == 0 ? 1u : static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / block_elems;
        const uint64_t bounded = std::max<uint64_t>(1, by_values);
        return static_cast<uint32_t>(
            std::min<uint64_t>(bounded, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
    }

    static uint32_t clamp_page_size(uint32_t requested, int uniform_block_size) {
        if (requested == 0) {
            return page_size_limit(uniform_block_size);
        }
        if (uniform_block_size <= 0) {
            return std::max<uint32_t>(requested, 1u);
        }
        return static_cast<uint32_t>(std::clamp<uint64_t>(
            requested,
            1u,
            static_cast<uint64_t>(page_size_limit(uniform_block_size))));
    }

    int block_size = 0;
    PagedBuffer<T> values;
    uint32_t page_setting_ = std::numeric_limits<uint32_t>::max();

    BSRMatrixBackend() = default;

    BSRMatrixBackend(int uniform_block_size, uint32_t page_size)
        : block_size(uniform_block_size),
          page_setting_(clamp_page_size(page_size, uniform_block_size)),
          values(std::max<uint32_t>(
              page_setting_ *
                  static_cast<uint32_t>(uniform_block_size * uniform_block_size),
              1u)) {}

    uint32_t page_setting() const {
        return page_setting_;
    }

    void set_page_setting(uint32_t page_size) {
        page_setting_ = clamp_page_size(page_size, block_size);
    }

    uint32_t page_size() const {
        return static_cast<uint32_t>(
            values.elements_per_page() / std::max<size_t>(block_value_count(), 1));
    }

    uint64_t value_count() const {
        return values.size();
    }

    size_t block_value_count() const {
        return static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
    }

    size_t block_count() const {
        const size_t values_per_block = block_value_count();
        return values_per_block == 0 ? 0 : static_cast<size_t>(values.size()) / values_per_block;
    }

    void initialize_structure(uint64_t logical_blocks, int uniform_block_size) {
        block_size = uniform_block_size;
        page_setting_ = clamp_page_size(page_setting_, block_size);
        rebuild_pages(logical_blocks);
        values.resize(logical_blocks * static_cast<uint64_t>(block_value_count()));
    }

    void initialize_structure(uint64_t logical_blocks, int uniform_block_size, uint32_t page_size) {
        block_size = uniform_block_size;
        set_page_setting(page_size);
        initialize_structure(logical_blocks, uniform_block_size);
    }

    void initialize_structure(const std::vector<int>& col_ind, int uniform_block_size) {
        initialize_structure(static_cast<uint64_t>(col_ind.size()), uniform_block_size);
    }

    void initialize_structure(const std::vector<int>& col_ind, int uniform_block_size, uint32_t page_size) {
        initialize_structure(static_cast<uint64_t>(col_ind.size()), uniform_block_size, page_size);
    }

    T* block_ptr(int slot) {
        return values.element_ptr(block_element_offset(static_cast<uint64_t>(slot)));
    }

    const T* block_ptr(int slot) const {
        return values.element_ptr(block_element_offset(static_cast<uint64_t>(slot)));
    }

    BSRPageSlice<T> page(const std::vector<int>& col_ind, uint32_t page_index) {
        auto value_page = values.page(page_index);
        const uint64_t first_block = first_block_from_page(value_page.first_element);
        const uint32_t block_count = blocks_in_page(value_page.count);
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRPageSlice<T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(block_value_count()),
            page_index,
            first_block};
    }

    BSRPageSlice<const T> page(const std::vector<int>& col_ind, uint32_t page_index) const {
        auto value_page = values.page(page_index);
        const uint64_t first_block = first_block_from_page(value_page.first_element);
        const uint32_t block_count = blocks_in_page(value_page.count);
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRPageSlice<const T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(block_value_count()),
            page_index,
            first_block};
    }

    template <typename Fn>
    void for_each_page(const std::vector<int>& col_ind, Fn&& fn) {
        for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
            fn(page(col_ind, page_index));
        }
    }

    template <typename Fn>
    void for_each_page(const std::vector<int>& col_ind, Fn&& fn) const {
        for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
            fn(page(col_ind, page_index));
        }
    }

    template <typename Fn>
    void for_each_row_slice(const std::vector<int>& row_ptr, const std::vector<int>& col_ind, int row, Fn&& fn) const {
        auto [current, end] = row_block_bounds(row_ptr, row);
        const uint32_t blocks_per_page = page_size();
        while (current < end) {
            const uint32_t page_index = static_cast<uint32_t>(current / blocks_per_page);
            const uint32_t local_offset = static_cast<uint32_t>(current % blocks_per_page);
            const auto page_slice = page(col_ind, page_index);
            const uint32_t block_count = static_cast<uint32_t>(
                std::min<uint64_t>(page_slice.block_count - local_offset, end - current));
            fn(trim_page_slice(page_slice, local_offset, block_count));
            current += block_count;
        }
    }

private:
    uint32_t active_page_size(uint64_t logical_blocks) const {
        if (logical_blocks == 0) {
            return page_setting_;
        }
        return static_cast<uint32_t>(std::min<uint64_t>(logical_blocks, page_setting_));
    }

    void rebuild_pages(uint64_t logical_blocks) {
        values = PagedBuffer<T>(
            std::max<uint32_t>(active_page_size(logical_blocks) * static_cast<uint32_t>(block_value_count()), 1u));
    }

    static std::pair<uint64_t, uint64_t> row_block_bounds(const std::vector<int>& row_ptr, int row) {
        return {
            static_cast<uint64_t>(row_ptr[static_cast<size_t>(row)]),
            static_cast<uint64_t>(row_ptr[static_cast<size_t>(row) + 1])};
    }

    uint64_t block_element_offset(uint64_t block_index) const {
        return block_index * static_cast<uint64_t>(block_value_count());
    }

    uint64_t first_block_from_page(uint64_t first_element) const {
        return first_element / static_cast<uint64_t>(std::max<size_t>(block_value_count(), 1));
    }

    uint32_t blocks_in_page(uint32_t element_count) const {
        return static_cast<uint32_t>(element_count / std::max<size_t>(block_value_count(), 1));
    }

    template <typename ValueType>
    static BSRPageSlice<ValueType> trim_page_slice(
        BSRPageSlice<ValueType> page_slice,
        uint32_t local_block_offset,
        uint32_t block_count) {
        if (local_block_offset > page_slice.block_count ||
            block_count > page_slice.block_count - local_block_offset) {
            throw std::out_of_range("BSRMatrixBackend::trim_page_slice range out of bounds");
        }
        return BSRPageSlice<ValueType>{
            page_slice.cols + local_block_offset,
            page_slice.values + static_cast<size_t>(local_block_offset) * page_slice.block_value_count,
            block_count,
            page_slice.block_size,
            page_slice.block_value_count,
            page_slice.page_index,
            page_slice.first_block + local_block_offset};
    }
};

template <typename T, typename Kernel>
struct VBCSRMatrixBackend {
    using Storage = ShapeBlockStore<T>;
    // The outer VBCSR backend still uses "slot" for the flat graph nonzero-block
    // index. The storage layer below is block-oriented and distinguishes matrix
    // block indices, shape block indices, and page block indices.

    static constexpr uint32_t hard_safe_slots_per_page() {
        return Storage::hard_safe_blocks_per_page();
    }

    static uint32_t normalize_configured_max_slots_per_page(uint32_t requested) {
        if (requested == 0) {
            return hard_safe_slots_per_page();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(requested, 1u, static_cast<uint64_t>(hard_safe_slots_per_page())));
    }

    enum class ExecutionKind {
        StaticFallback,
        BatchedFallback,
        JIT
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

    struct ShapeBatchView {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        int page_id = -1;
        typename Storage::ShapePage page{};
        const ShapeExecutionPolicy* policy = nullptr;

        uint32_t block_count() const {
            return page.block_count;
        }

        uint32_t block_capacity() const {
            return page.blocks_per_page;
        }

        int logical_slot(uint32_t block_index) const {
            return page.matrix_block(block_index);
        }

        const T* block_ptr(uint32_t block_index) const {
            return page.block_ptr(block_index);
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
    explicit VBCSRMatrixBackend(uint32_t max_slots_per_page)
        : storage(max_slots_per_page) {}
    VBCSRMatrixBackend(const VBCSRMatrixBackend&) = delete;
    VBCSRMatrixBackend& operator=(const VBCSRMatrixBackend&) = delete;

    VBCSRMatrixBackend(VBCSRMatrixBackend&& other) noexcept
        : blk_handles(std::move(other.blk_handles)),
          storage(std::move(other.storage)),
          contiguous_layout(other.contiguous_layout),
          shape_policies(std::move(other.shape_policies)),
          spmm_policy_lookup(std::move(other.spmm_policy_lookup)),
          spmm_policy_records(std::move(other.spmm_policy_records)) {}

    VBCSRMatrixBackend& operator=(VBCSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            blk_handles = std::move(other.blk_handles);
            storage = std::move(other.storage);
            contiguous_layout = other.contiguous_layout;
            shape_policies = std::move(other.shape_policies);
            std::lock_guard<std::mutex> lock(policy_mutex);
            spmm_policy_lookup = std::move(other.spmm_policy_lookup);
            spmm_policy_records = std::move(other.spmm_policy_records);
        }
        return *this;
    }

    // Per-logical-slot payload handle. The slot index is the flat local nonzero-block
    // index from row_ptr/col_ind; the handle locates that block's shape-page payload.
    std::vector<uint64_t> blk_handles;
    Storage storage;
    bool contiguous_layout = false;
    std::vector<std::unique_ptr<ShapeExecutionPolicy>> shape_policies;
    mutable std::mutex policy_mutex;
    mutable std::map<SpMMPolicyKey, size_t> spmm_policy_lookup;
    mutable std::vector<std::unique_ptr<SpMMExecutionPolicy>> spmm_policy_records;

    size_t local_scalar_nnz() const {
        return storage.scalar_value_count();
    }

    int shape_class_count() const {
        return storage.shape_count();
    }

    uint32_t configured_max_slots_per_page() const {
        return storage.max_blocks_per_page();
    }

    void set_configured_max_slots_per_page(uint32_t max_slots_per_page) {
        storage.set_max_blocks_per_page(max_slots_per_page);
    }

    int ensure_shape(int row_dim, int col_dim, size_t expected_block_count = 0) {
        const int shape_id = storage.get_or_create_shape(row_dim, col_dim, expected_block_count);
        ensure_shape_policy(shape_id);
        return shape_id;
    }

    uint64_t append_block_for_shape(int shape_id, int logical_slot) {
        return storage.append(shape_id, logical_slot);
    }

    void rebuild_handle_table() {
        storage.rebuild_handles(blk_handles);
    }

    ExecutionKind execution_kind_for_shape(int shape_id) const {
        const auto* policy = shape_policy(shape_id);
        return policy == nullptr
            ? ExecutionKind::StaticFallback
            : policy->preferred_execution.load(std::memory_order_relaxed);
    }

    ExecutionKind execution_kind_for_spmm_triple(int row_dim, int inner_dim, int col_dim) const {
        const auto* policy = ensure_spmm_policy(row_dim, inner_dim, col_dim);
        if (policy == nullptr) {
            return ExecutionKind::StaticFallback;
        }
        return policy->preferred_execution.load(std::memory_order_relaxed);
    }

    void record_apply_batch(int shape_id, size_t block_count) const {
        auto* policy = shape_policy(shape_id);
        if (policy == nullptr) {
            return;
        }
        auto& entry = *policy;
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
        auto* policy = shape_policy(shape_id);
        return policy == nullptr
            ? 0
            : static_cast<size_t>(policy->apply_batches.load(std::memory_order_relaxed));
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
        (void)graph;
        (void)row_ptr;
        (void)col_ind;
        storage.rebuild_handles(blk_handles);
        contiguous_layout = true;
    }

    template <typename Fn>
    void for_each_shape_class(Fn&& fn) const {
        storage.for_each_shape([&](const auto& record) {
            fn(record.shape_id, record.row_dim, record.col_dim, record.matrix_block_indices);
        });
    }

    template <typename Fn>
    void for_each_shape_batch(Fn&& fn) const {
        storage.for_each_page([&](const typename Storage::ShapePage& page) {
            ShapeBatchView view{
                page.shape_id,
                page.row_dim,
                page.col_dim,
                page.page_id,
                page,
                shape_policy(page.shape_id)};
            if constexpr (std::is_invocable_v<Fn, const ShapeBatchView&>) {
                fn(view);
            } else {
                std::vector<int> matrix_block_indices;
                matrix_block_indices.reserve(page.block_count);
                for (uint32_t idx = 0; idx < page.block_count; ++idx) {
                    matrix_block_indices.push_back(page.matrix_block(idx));
                }
                fn(page.shape_id, page.row_dim, page.col_dim, page.page_id, matrix_block_indices);
            }
        });
    }

    T* get_ptr(uint64_t handle) {
        return storage.block_ptr(handle);
    }

    const T* get_ptr(uint64_t handle) const {
        return storage.block_ptr(handle);
    }

    size_t block_size_elements(int slot) const {
        return storage.elements_per_block(
            Storage::shape_id_of(blk_handles.at(static_cast<size_t>(slot))));
    }

private:
    ShapeExecutionPolicy* ensure_shape_policy(int shape_id) {
        if (shape_id < 0) {
            throw std::logic_error("Negative VBCSR shape id");
        }
        if (static_cast<size_t>(shape_id) >= shape_policies.size()) {
            shape_policies.resize(static_cast<size_t>(shape_id) + 1);
        }
        if (!shape_policies[static_cast<size_t>(shape_id)]) {
            shape_policies[static_cast<size_t>(shape_id)] = std::make_unique<ShapeExecutionPolicy>();
        }
        return shape_policies[static_cast<size_t>(shape_id)].get();
    }

    ShapeExecutionPolicy* shape_policy(int shape_id) const {
        if (shape_id < 0 || static_cast<size_t>(shape_id) >= shape_policies.size()) {
            return nullptr;
        }
        return shape_policies[static_cast<size_t>(shape_id)].get();
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
