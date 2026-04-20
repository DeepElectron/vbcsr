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

// What backend in charge
// 1. storage
// 2. page/batch/slice
// 3. vendor

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

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_BACKEND_COMMON_HPP
