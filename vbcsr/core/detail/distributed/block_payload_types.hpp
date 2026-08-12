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

// An owning block record: block_csr's assembly staging builds these to SEND.
template <typename T>
struct FetchedBlock {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    std::vector<T> data;
};

// A fetched block as the payload exchange RETURNS it: a view, not an owner.
// Remote payloads live in the context's arena below; blocks the rank already
// owned point straight into the source matrix's storage (no copy at all), so
// the SOURCE MATRIX MUST OUTLIVE THE CONTEXT -- true of every caller, which
// holds the operand it is fetching from for the duration of its product.
template <typename T>
struct FetchedBlockRef {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    const T* data;
};

template <typename T>
struct FetchedBlockContext {
    std::vector<FetchedBlockRef<T>> blocks;
    std::map<int, int> row_sizes;
    // One allocation owning every REMOTE payload the refs point into. This
    // replaces one heap vector per fetched block -- at halo scale, tens of
    // millions of allocations and a serial copy loop -- with a single sized
    // arena filled by a parallel copy.
    std::vector<T> arena;
};

} // namespace detail

} // namespace vbcsr

#endif // VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP
