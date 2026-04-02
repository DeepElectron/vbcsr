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
    explicit CSRResultBuilder(DistGraph* graph, uint32_t nnz_per_page = 0) : graph_(graph) {
        if (graph_ == nullptr) {
            return;
        }
        backend_.initialize_structure(graph_->adj_ind, nnz_per_page);
    }

    void accumulate_block(int slot, const T* src, T alpha = T(1)) {
        *backend_.value_ptr(slot) += alpha * (*src);
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

    CSRMatrixBackend<T> commit() && {
        return std::move(backend_);
    }

    T* slot_data(int slot) {
        return backend_.value_ptr(slot);
    }

    const T* slot_data(int slot) const {
        return backend_.value_ptr(slot);
    }

private:
    DistGraph* graph_ = nullptr;
    CSRMatrixBackend<T> backend_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_CSR_RESULT_BUILDER_HPP
