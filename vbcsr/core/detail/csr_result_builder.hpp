#ifndef VBCSR_DETAIL_CSR_RESULT_BUILDER_HPP
#define VBCSR_DETAIL_CSR_RESULT_BUILDER_HPP

#include "backend_handle.hpp"
#include "../dist_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

template <typename T>
class CSRResultBuilder {
public:
    explicit CSRResultBuilder(DistGraph* graph) : graph_(graph) {
        if (graph_ == nullptr) {
            return;
        }

        const size_t nnz = graph_->adj_ind.size();
        blk_handles_.resize(nnz);
        blk_sizes_.assign(nnz, 1);
        arena_.reserve(nnz);
        for (size_t slot = 0; slot < nnz; ++slot) {
            blk_handles_[slot] = arena_.allocate(1);
        }
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
            throw std::runtime_error("CSRResultBuilder could not locate destination slot");
        }
        return static_cast<int>(std::distance(graph_->adj_ind.begin(), it));
    }

    CSRMatrixBackend<T> commit_backend() && {
        CSRMatrixBackend<T> backend;
        backend.blk_handles = std::move(blk_handles_);
        backend.blk_sizes = std::move(blk_sizes_);
        backend.arena = std::move(arena_);
        return backend;
    }

private:
    DistGraph* graph_ = nullptr;
    std::vector<uint64_t> blk_handles_;
    std::vector<size_t> blk_sizes_;
    BlockArena<T> arena_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_CSR_RESULT_BUILDER_HPP
