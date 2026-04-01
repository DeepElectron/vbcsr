#ifndef VBCSR_DETAIL_LEGACY_MATRIX_BACKEND_HPP
#define VBCSR_DETAIL_LEGACY_MATRIX_BACKEND_HPP

#include "../block_memory_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vbcsr::detail {

template <typename T>
struct LegacyMatrixBackend {
    std::vector<uint64_t> blk_handles;
    std::vector<size_t> blk_sizes;
    BlockArena<T> arena;

    size_t local_scalar_nnz() const {
        size_t total = 0;
        for (size_t size : blk_sizes) {
            total += size;
        }
        return total;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_LEGACY_MATRIX_BACKEND_HPP
