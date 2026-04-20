#ifndef VBCSR_DETAIL_BACKEND_BACKEND_COMMON_HPP
#define VBCSR_DETAIL_BACKEND_BACKEND_COMMON_HPP

#include "../storage/paged_array.hpp"
#include "../storage/shape_paged_storage.hpp"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(VBCSR_HAVE_MKL_SPARSE) || defined(VBCSR_HAVE_MKL_BSR_SPARSE)
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

struct PageRowSpan {
    int row_begin = 0;
    int row_end = 0;

    int row_count() const {
        return row_end - row_begin;
    }
};

inline PageRowSpan find_page_row_span(
    const std::vector<int>& row_ptr,
    int page_begin,
    int page_end) {
    const auto row_begin_it =
        std::upper_bound(row_ptr.begin() + 1, row_ptr.end(), page_begin);
    const auto row_end_it =
        std::lower_bound(row_ptr.begin(), row_ptr.end() - 1, page_end);
    return PageRowSpan{
        static_cast<int>(std::distance(row_ptr.begin(), row_begin_it)) - 1,
        static_cast<int>(std::distance(row_ptr.begin(), row_end_it))};
} // so this function is implemented to find the index of the start and end of the page.
// the implementation is really bad, very hard to understand.

template <typename EmitFn>
void emit_page_local_row_ptr(
    const std::vector<int>& row_ptr,
    int page_begin,
    int page_end,
    PageRowSpan row_span,
    EmitFn&& emit) {
    // Pages may cut through the first and last rows. Their page-local row_ptr
    // entries are therefore fixed by the page boundaries; only interior row
    // starts need to be translated from global block/NNZ offsets.
    emit(0);
    for (int row = row_span.row_begin + 1; row < row_span.row_end; ++row) {
        emit(row_ptr[static_cast<size_t>(row)] - page_begin);
    }
    emit(page_end - page_begin);
}

#if defined(VBCSR_HAVE_MKL_SPARSE) || defined(VBCSR_HAVE_MKL_BSR_SPARSE)
inline void destroy_mkl_sparse_handle(sparse_matrix_t& handle) {
    if (handle != nullptr) {
        mkl_sparse_destroy(handle);
        handle = nullptr;
    }
}

inline matrix_descr make_mkl_descr() {
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    return descr;
}

template <typename T>
constexpr sparse_operation_t mkl_adjoint_operation() {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }
    return SPARSE_OPERATION_TRANSPOSE;
}

template <typename Tag>
struct SparseVendorMKLMMVariant {
    int num_rhs = -1;
    uint64_t last_use = 0;
    sparse_matrix_t handle = nullptr;

    SparseVendorMKLMMVariant() = default;
    SparseVendorMKLMMVariant(const SparseVendorMKLMMVariant&) = delete;
    SparseVendorMKLMMVariant& operator=(const SparseVendorMKLMMVariant&) = delete;

    SparseVendorMKLMMVariant(SparseVendorMKLMMVariant&& other) noexcept
        : num_rhs(other.num_rhs),
          last_use(other.last_use),
          handle(other.handle) {
        other.num_rhs = -1;
        other.last_use = 0;
        other.handle = nullptr;
    }

    SparseVendorMKLMMVariant& operator=(SparseVendorMKLMMVariant&& other) noexcept {
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

    ~SparseVendorMKLMMVariant() {
        reset();
    }

    void reset() {
        destroy_mkl_sparse_handle(handle);
        num_rhs = -1;
        last_use = 0;
    }
};

template <typename Variant>
struct SparseVendorMKLHandle {
    using MMVariant = Variant;

    static constexpr size_t kMaxMMVariants = 4;
    static constexpr size_t kInvalidMMVariantIndex = static_cast<size_t>(-1);

    sparse_matrix_t mv_handle = nullptr;
    std::vector<Variant> mm_variants;
    uint64_t use_clock = 0;
    size_t recent_mm_variant_index = kInvalidMMVariantIndex;

    SparseVendorMKLHandle() = default;
    SparseVendorMKLHandle(const SparseVendorMKLHandle&) = delete;
    SparseVendorMKLHandle& operator=(const SparseVendorMKLHandle&) = delete;

    SparseVendorMKLHandle(SparseVendorMKLHandle&& other) noexcept
        : mv_handle(other.mv_handle),
          mm_variants(std::move(other.mm_variants)),
          use_clock(other.use_clock),
          recent_mm_variant_index(other.recent_mm_variant_index) {
        other.mv_handle = nullptr;
        other.use_clock = 0;
        other.recent_mm_variant_index = kInvalidMMVariantIndex;
    }

    SparseVendorMKLHandle& operator=(SparseVendorMKLHandle&& other) noexcept {
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

    ~SparseVendorMKLHandle() {
        reset();
    }

    void reset() {
        destroy_mkl_sparse_handle(mv_handle);
        mm_variants.clear();
        use_clock = 0;
        recent_mm_variant_index = kInvalidMMVariantIndex;
    }

    Variant* find_mm_variant(int num_rhs) {
        if (recent_mm_variant_index < mm_variants.size() &&
            mm_variants[recent_mm_variant_index].num_rhs == num_rhs) {
            return &mm_variants[recent_mm_variant_index];
        }
        for (auto& variant : mm_variants) {
            if (variant.num_rhs == num_rhs) {
                recent_mm_variant_index =
                    static_cast<size_t>(&variant - mm_variants.data());
                return &variant;
            }
        }
        return nullptr;
    }

    const Variant* find_mm_variant(int num_rhs) const {
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

template <typename MKLHandle, typename BuildFn>
bool ensure_mkl_mm_variant_with_lru( // LRU: Least recently used
    MKLHandle& handle,
    int num_rhs,
    BuildFn&& build_variant) {
    if (auto* existing = handle.find_mm_variant(num_rhs)) {
        existing->last_use = handle.next_use_stamp();
        return true;
    }

    typename MKLHandle::MMVariant variant;
    if (!build_variant(variant)) {
        return false;
    }
    variant.last_use = handle.next_use_stamp();

    // if there's still room, add without replacement
    if (handle.mm_variants.size() < MKLHandle::kMaxMMVariants) {
        handle.mm_variants.push_back(std::move(variant));
        handle.remember_mm_variant(handle.mm_variants.size() - 1);
        return true;
    }

    // if full, replace the least recently used variant
    size_t lru_index = 0;
    for (size_t idx = 1; idx < handle.mm_variants.size(); ++idx) {
        if (handle.mm_variants[idx].last_use <
            handle.mm_variants[lru_index].last_use) {
            lru_index = idx;
        }
    }
    handle.mm_variants[lru_index] = std::move(variant);
    handle.remember_mm_variant(lru_index);
    return true;
}
#endif

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_BACKEND_COMMON_HPP
