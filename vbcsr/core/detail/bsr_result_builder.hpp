#ifndef VBCSR_DETAIL_BSR_RESULT_BUILDER_HPP
#define VBCSR_DETAIL_BSR_RESULT_BUILDER_HPP

#include "backend_handle.hpp"
#include "../dist_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

template <typename T>
class BSRResultBuilder {
public:
    explicit BSRResultBuilder(DistGraph* graph)
        : graph_(graph), block_size_(infer_block_size(graph)) {
        if (graph_ == nullptr) {
            return;
        }

        const size_t nnz = graph_->adj_ind.size();
        const size_t block_area = static_cast<size_t>(block_size_) * static_cast<size_t>(block_size_);
        blk_handles_.resize(nnz);
        blk_sizes_.assign(nnz, block_area);
        arena_.reserve(nnz * block_area);
        for (size_t slot = 0; slot < nnz; ++slot) {
            blk_handles_[slot] = arena_.allocate(block_area);
        }
    }

    static int infer_block_size(const DistGraph* graph) {
        if (graph == nullptr || graph->block_sizes.empty()) {
            return 0;
        }

        int block_size = 0;
        for (int dim : graph->block_sizes) {
            if (dim <= 1) {
                throw std::runtime_error("BSRResultBuilder requires uniform block sizes greater than 1");
            }
            if (block_size == 0) {
                block_size = dim;
                continue;
            }
            if (dim != block_size) {
                throw std::runtime_error("BSRResultBuilder requires a uniform block size");
            }
        }
        return block_size;
    }

    int block_size() const {
        return block_size_;
    }

    T* slot_data(int slot) {
        return arena_.get_ptr(blk_handles_[slot]);
    }

    const T* slot_data(int slot) const {
        return arena_.get_ptr(blk_handles_[slot]);
    }

    int find_slot(int local_row, int local_col) const {
        const int start = graph_->adj_ptr[local_row];
        const int end = graph_->adj_ptr[local_row + 1];
        auto begin = graph_->adj_ind.begin() + start;
        auto finish = graph_->adj_ind.begin() + end;
        auto it = std::lower_bound(begin, finish, local_col);
        if (it == finish || *it != local_col) {
            throw std::runtime_error("BSRResultBuilder could not locate destination slot");
        }
        return static_cast<int>(std::distance(graph_->adj_ind.begin(), it));
    }

    BSRMatrixBackend<T> commit_backend() && {
        BSRMatrixBackend<T> backend;
        backend.block_size = block_size_;
        backend.blk_handles = std::move(blk_handles_);
        backend.blk_sizes = std::move(blk_sizes_);
        backend.arena = std::move(arena_);
        return backend;
    }

private:
    DistGraph* graph_ = nullptr;
    int block_size_ = 0;
    std::vector<uint64_t> blk_handles_;
    std::vector<size_t> blk_sizes_;
    BlockArena<T> arena_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BSR_RESULT_BUILDER_HPP
