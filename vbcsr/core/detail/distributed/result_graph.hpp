#ifndef VBCSR_DETAIL_DISTRIBUTED_RESULT_GRAPH_HPP
#define VBCSR_DETAIL_DISTRIBUTED_RESULT_GRAPH_HPP

#include "../../dist_graph.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbcsr::detail {

template <typename GhostSizeMap>
void backfill_ghost_block_sizes(DistGraph& graph, const GhostSizeMap& ghost_sizes, const char* context) {
    const int n_cols = static_cast<int>(graph.global_to_local.size());
    if (static_cast<int>(graph.block_sizes.size()) < n_cols) {
        graph.block_sizes.resize(n_cols, 0);
    }

    #pragma omp parallel for
    for (int i = 0; i < n_cols; ++i) {
        if (graph.block_sizes[i] > 0) {
            continue;
        }

        const int global_col = graph.get_global_index(i);
        auto it = ghost_sizes.find(global_col);
        if (it == ghost_sizes.end()) {
            throw std::runtime_error(std::string("Missing block size while building ") + context + " result graph");
        }
        graph.block_sizes[i] = it->second;
    }

    graph.block_offsets.resize(n_cols + 1);
    graph.block_offsets[0] = 0;
    for (int i = 0; i < n_cols; ++i) {
        graph.block_offsets[i + 1] = graph.block_offsets[i] + graph.block_sizes[i];
    }
}

inline std::vector<int> owned_block_sizes(const DistGraph& graph) {
    const size_t n_owned = graph.owned_global_indices.size();
    return std::vector<int>(graph.block_sizes.begin(), graph.block_sizes.begin() + static_cast<long long>(n_owned));
}

template <typename GhostSizeMap>
DistGraph* construct_result_graph(
    MPI_Comm comm,
    const std::vector<int>& owned_global_indices,
    const std::vector<int>& owned_block_sizes,
    const std::vector<std::vector<int>>& adjacency,
    const GhostSizeMap& ghost_sizes,
    const char* context) {
    auto* graph = new DistGraph(comm);
    if (graph->size == 1) {
        const int n_owned = static_cast<int>(owned_global_indices.size());
        graph->owned_global_indices = owned_global_indices;
        graph->ghost_global_indices.clear();
        graph->global_to_local.clear();
        for (int idx = 0; idx < n_owned; ++idx) {
            graph->global_to_local[owned_global_indices[static_cast<size_t>(idx)]] = idx;
        }

        graph->block_sizes = owned_block_sizes;
        graph->block_displs.assign({0, n_owned});
        graph->adj_ptr.assign(static_cast<size_t>(n_owned) + 1, 0);
        std::vector<int> staged_adj_ind;
        std::vector<int> row_cols;
        for (int row = 0; row < n_owned; ++row) {
            row_cols.clear();
            row_cols.reserve(adjacency[static_cast<size_t>(row)].size());
            for (int global_col : adjacency[static_cast<size_t>(row)]) {
                auto it = graph->global_to_local.find(global_col);
                if (it == graph->global_to_local.end()) {
                    throw std::runtime_error(std::string("Invalid serial ") + context + " result graph column");
                }
                row_cols.push_back(it->second);
            }
            std::sort(row_cols.begin(), row_cols.end());
            staged_adj_ind.insert(staged_adj_ind.end(), row_cols.begin(), row_cols.end());
            graph->adj_ptr[static_cast<size_t>(row) + 1] =
                static_cast<int>(staged_adj_ind.size());
        }
        if (staged_adj_ind.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "result adjacency exceeds 2^31 blocks on this rank; distribute over more ranks");
        }
        graph->adj_ind.assign(staged_adj_ind.data(), staged_adj_ind.size());

        graph->block_offsets.resize(graph->block_sizes.size() + 1);
        graph->block_offsets[0] = 0;
        for (size_t idx = 0; idx < graph->block_sizes.size(); ++idx) {
            graph->block_offsets[idx + 1] =
                graph->block_offsets[idx] + graph->block_sizes[idx];
        }

        graph->send_counts.assign(1, 0);
        graph->recv_counts.assign(1, 0);
        graph->send_indices.clear();
        graph->recv_indices.clear();
        graph->send_displs.assign(2, 0);
        graph->recv_displs.assign(2, 0);
        graph->send_ranks.clear();
        graph->recv_ranks.clear();
        graph->send_counts_scalar.assign(1, 0);
        graph->recv_counts_scalar.assign(1, 0);
        graph->send_displs_scalar.assign(2, 0);
        graph->recv_displs_scalar.assign(2, 0);
        return graph;
    }
    graph->construct_distributed(owned_global_indices, owned_block_sizes, adjacency);
    backfill_ghost_block_sizes(*graph, ghost_sizes, context);
    return graph;
}

template <typename Matrix, typename GhostSizeMap>
DistGraph* construct_result_graph(
    const Matrix& matrix,
    const std::vector<std::vector<int>>& adjacency,
    const GhostSizeMap& ghost_sizes,
    const char* context) {
    return construct_result_graph(
        matrix.graph->comm,
        matrix.graph->owned_global_indices,
        owned_block_sizes(*matrix.graph),
        adjacency,
        ghost_sizes,
        context);
}

} // namespace vbcsr::detail

#endif
