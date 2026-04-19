#ifndef VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP
#define VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP

#include <map>
#include <vector>

namespace vbcsr {

struct BlockID {
    int row;
    int col;

    bool operator<(const BlockID& other) const {
        if (row != other.row) {
            return row < other.row;
        }
        return col < other.col;
    }
};

namespace detail {

template <typename T>
struct FetchedBlock {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    std::vector<T> data;
};

template <typename T>
struct FetchedBlockContext {
    std::vector<FetchedBlock<T>> blocks;
    std::map<int, int> row_sizes;
};

} // namespace detail

} // namespace vbcsr

#endif // VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP
