#ifndef VBCSR_DETAIL_DISTRIBUTED_RESULT_GRAPH_HPP
#define VBCSR_DETAIL_DISTRIBUTED_RESULT_GRAPH_HPP

#include "../../dist_graph.hpp"

#include <algorithm>
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
