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
    explicit BSRResultBuilder(DistGraph* graph, uint32_t blocks_per_page = 0)
        : graph_(graph), block_size_(infer_block_size(graph)) {
        if (graph_ == nullptr) {
            return;
        }
        backend_.initialize_structure(graph_->adj_ind, block_size_, blocks_per_page);
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

    void accumulate_block(int slot, const T* src, T alpha = T(1)) {
        T* dest = backend_.block_ptr(slot);
        const size_t block_area = static_cast<size_t>(block_size_) * static_cast<size_t>(block_size_);
        for (size_t idx = 0; idx < block_area; ++idx) {
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
            throw std::runtime_error("BSRResultBuilder could not locate destination slot");
        }
        return static_cast<int>(std::distance(graph_->adj_ind.begin(), it));
    }

    BSRMatrixBackend<T> commit() && {
        return std::move(backend_);
    }

    T* slot_data(int slot) {
        return backend_.block_ptr(slot);
    }

    const T* slot_data(int slot) const {
        return backend_.block_ptr(slot);
    }

private:
    DistGraph* graph_ = nullptr;
    int block_size_ = 0;
    BSRMatrixBackend<T> backend_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BSR_RESULT_BUILDER_HPP
