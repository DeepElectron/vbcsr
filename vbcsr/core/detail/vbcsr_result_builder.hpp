#ifndef VBCSR_DETAIL_VBCSR_RESULT_BUILDER_HPP
#define VBCSR_DETAIL_VBCSR_RESULT_BUILDER_HPP

#include "backend_handle.hpp"
#include "../dist_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

template <typename T, typename Kernel>
class VBCSRResultBuilder {
public:
    explicit VBCSRResultBuilder(DistGraph* graph) : graph_(graph) {
        if (graph_ == nullptr) {
            return;
        }

        const size_t nnz = graph_->adj_ind.size();
        backend_.blk_handles.resize(nnz);
        backend_.slot_shape_ids.resize(nnz, -1);
        const int n_rows = static_cast<int>(graph_->owned_global_indices.size());
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph_->block_sizes[row];
            for (int slot = graph_->adj_ptr[row]; slot < graph_->adj_ptr[row + 1]; ++slot) {
                const int col = graph_->adj_ind[slot];
                const int col_dim = graph_->block_sizes[col];
                const int shape_id = backend_.ensure_shape(row_dim, col_dim);
                const uint64_t handle = backend_.allocate_slot_for_shape(shape_id);
                backend_.bind_logical_slot(slot, shape_id, handle);
            }
        }
        backend_.rebuild_shape_registry(graph_, graph_->adj_ptr, graph_->adj_ind);
    }

    T* mutable_block(int slot) {
        return backend_.get_ptr(backend_.blk_handles[slot]);
    }

    const T* mutable_block(int slot) const {
        return backend_.get_ptr(backend_.blk_handles[slot]);
    }

    void accumulate_block(int slot, const T* src, T alpha = T(1)) {
        T* dest = mutable_block(slot);
        const size_t size = backend_.block_size_elements(slot);
        for (size_t idx = 0; idx < size; ++idx) {
            dest[idx] += alpha * src[idx];
        }
    }

    int find_slot(int local_row, int local_col) const {
        const int start = graph_->adj_ptr[local_row];
        const int end = graph_->adj_ptr[local_row + 1];
        auto begin = graph_->adj_ind.begin() + start;
        auto finish = graph_->adj_ind.begin() + end;
        auto it = std::lower_bound(begin, finish, local_col);
        if (it == finish || *it != local_col) {
            throw std::runtime_error("VBCSRResultBuilder could not locate destination slot");
        }
        return static_cast<int>(std::distance(graph_->adj_ind.begin(), it));
    }

    VBCSRMatrixBackend<T, Kernel> commit() && {
        backend_.rebuild_shape_registry(graph_, graph_->adj_ptr, graph_->adj_ind);
        return std::move(backend_);
    }

    T* slot_data(int slot) {
        return mutable_block(slot);
    }

    const T* slot_data(int slot) const {
        return mutable_block(slot);
    }

    VBCSRMatrixBackend<T, Kernel> commit_backend() && {
        return std::move(*this).commit();
    }

private:
    DistGraph* graph_ = nullptr;
    VBCSRMatrixBackend<T, Kernel> backend_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_VBCSR_RESULT_BUILDER_HPP
