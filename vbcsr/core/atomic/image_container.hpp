#pragma once
#include "atomic_data.hpp"
#include "../block_csr.hpp"
#include <map>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

namespace vbcsr {
namespace atomic {

enum class PhaseConvention {
    R_ONLY,          // exp(i * K * R)
    R_AND_POSITION   // exp(i * K * (R + r_j - r_i))
};

template <typename T, typename Kernel = DefaultKernel<T>>
class ImageContainer {
public:
    AtomicData* atom_data;
    DistGraph* base_graph; // Reference to atom_data->graph
    
    // Map R vector (rx, ry, rz) to its specific graph and matrix
    std::map<std::vector<int>, DistGraph*> image_graphs;
    std::map<std::vector<int>, BlockSpMat<T, Kernel>*> image_blocks;

    using ComplexT = std::complex<double>;
    // We might need a complex kernel for the result of sample_k if T is real
    // But usually T is already complex for Hamiltonians? 
    // If T is double, sample_k returns complex.
    // Let's assume for now we return BlockSpMat<std::complex<double>>.
    
    // Actually, if T is double, we can't store complex result in BlockSpMat<T>.
    // We need a separate type for the result.
    // Let's define ResultT as complex<double> always for now, or complex<real_part<T>>.
    
    using ResultT = std::complex<double>;
    using ResultKernel = DefaultKernel<ResultT>;

public:
    ImageContainer(AtomicData* data) : atom_data(data), base_graph(data->graph) {
        build_image_graphs();
    }

    ~ImageContainer() {
        for (auto& kv : image_graphs) {
            delete kv.second;
        }
        for (auto& kv : image_blocks) {
            delete kv.second;
        }
    }

    void build_image_graphs() {
        // 1. Identify all unique R vectors in edges
        // We need to scan all edges to find which R they belong to.
        // Edges in AtomicData are stored as: src (local), dst (local), rx, ry, rz.
        
        // We need to group edges by R.
        // Map: R -> adjacency list for that R
        // Adjacency list: src_lid -> list of dst_gid
        
        std::map<std::vector<int>, std::vector<std::vector<int>>> adj_by_r;
        
        int n_owned = atom_data->n_atom;
        
        // Initialize adj for known Rs? No, discover them.
        
        for (int i = 0; i < n_owned; ++i) {
            const auto& edges = atom_data->get_atom_edges(i);
            for (int edge_idx : edges) {
                int dst_lid = atom_data->get_edge_dst(edge_idx);
                int dst_gid = atom_data->get_global_index(dst_lid);
                
                int rx, ry, rz;
                atom_data->get_edge_shift_vec(edge_idx, &rx, &ry, &rz);
                std::vector<int> R = {rx, ry, rz};
                
                if (adj_by_r.find(R) == adj_by_r.end()) {
                    adj_by_r[R].resize(n_owned);
                }
                
                adj_by_r[R][i].push_back(dst_gid);
            }
        }
        
        // Ensure R=0 has diagonal blocks
        std::vector<int> R0 = {0, 0, 0};
        if (adj_by_r.find(R0) == adj_by_r.end()) {
            adj_by_r[R0].resize(n_owned);
        }
        for (int i = 0; i < n_owned; ++i) {
            int gid = atom_data->get_global_index(i);
            adj_by_r[R0][i].push_back(gid);
        }
        
        // Also need to ensure all ranks agree on the set of Rs?
        // DistGraph construction is collective. If a rank has no edges for a specific R, 
        // but other ranks do, it must still participate in DistGraph construction 
        // (passing empty adj is fine, but it needs to call the constructor).
        // So we need to gather all Rs from all ranks.
        
        // 1.1 Collect local Rs
        std::vector<int> local_Rs;
        for (auto& kv : adj_by_r) {
            local_Rs.push_back(kv.first[0]);
            local_Rs.push_back(kv.first[1]);
            local_Rs.push_back(kv.first[2]);
        }
        
        // 1.2 Allgather Rs across all ranks
        int size = atom_data->size;
        int my_count = local_Rs.size();
        std::vector<int> all_Rs;
        
        int initialized = 0;
        MPI_Initialized(&initialized);
        
        if (initialized && size > 1) {
            std::vector<int> recv_counts(size);
            MPI_Allgather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, atom_data->comm);
            
            std::vector<int> displs(size + 1, 0);
            for (int i = 0; i < size; ++i) displs[i+1] = displs[i] + recv_counts[i];
            
            all_Rs.resize(displs[size]);
            MPI_Allgatherv(local_Rs.data(), my_count, MPI_INT, all_Rs.data(), recv_counts.data(), displs.data(), MPI_INT, atom_data->comm);
        } else {
            // Serial: local Rs are the complete set
            all_Rs = local_Rs;
        }
        
        // 1.3 Unique Rs
        std::vector<std::vector<int>> unique_Rs;
        for (size_t i = 0; i < all_Rs.size(); i += 3) {
            unique_Rs.push_back({all_Rs[i], all_Rs[i+1], all_Rs[i+2]});
        }
        std::sort(unique_Rs.begin(), unique_Rs.end());
        unique_Rs.erase(std::unique(unique_Rs.begin(), unique_Rs.end()), unique_Rs.end());
        
        // 2. Construct graphs for each R
        std::vector<int> owned_indices(n_owned);
        for(int i=0; i<n_owned; ++i) owned_indices[i] = atom_data->get_global_index(i);
        
        std::vector<int> my_block_sizes(n_owned);
        for(int i=0; i<n_owned; ++i) {
            int norb;
            atom_data->get_atom_norb(i, &norb);
            my_block_sizes[i] = norb;
        }
        
        for (const auto& R : unique_Rs) {
            DistGraph* g = new DistGraph(atom_data->comm);
            
            // Get adj for this R (might be empty)
            std::vector<std::vector<int>> adj;
            if (adj_by_r.count(R)) {
                adj = adj_by_r[R];
            } else {
                adj.resize(n_owned);
            }
            
            // Remove duplicates in adj if any (AtomicData shouldn't have duplicate edges for same R, but good to be safe)
            for (auto& neighbors : adj) {
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }
            
            g->construct_distributed(owned_indices, my_block_sizes, adj);
            
            image_graphs[R] = g;
            
            // Also allocate the matrix
            BlockSpMat<T, Kernel>* mat = new BlockSpMat<T, Kernel>(g);
            // mat->owns_graph = false; // We manage graph lifetime
            image_blocks[R] = mat;
        }
    }

    void add_block(const std::vector<int>& R, int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = MatrixLayout::ColMajor) {
        auto it = image_blocks.find(R);
        if (it == image_blocks.end()) {
             // This should not happen if we built graphs for all Rs in AtomicData.
             // Unless the user is adding a block for an R that didn't exist in the original edges?
             // The requirement says: "if not, it should allocate the block_csr using the graph corresponding to that R."
             // But constructing a distributed graph is collective. We can't just do it on one rank.
             // If this R is new, we must assume all ranks call add_block or we have a mechanism to add new Rs collectively.
             // For now, let's assume R must exist or throw. 
             // Or, if the user meant "allocate the block in the matrix", that's handled by BlockSpMat.
             // If the user meant "create a new graph for a new R", that's complex.
             // Given "The base graph actually corresponding to a summation of graph for all shifted vector R",
             // it implies all possible edges are in base_graph (AtomicData).
             // So R must be in AtomicData.
             throw std::runtime_error("R vector not found in ImageContainer (must be present in AtomicData)");
        }
        
        it->second->add_block(global_row, global_col, data, rows, cols, mode, layout);
    }
    
    void assemble() {
        for (auto& kv : image_blocks) {
            kv.second->assemble();
        }
    }

    // Accumulate all image blocks onto a compatible reference graph using
    // a per-image weight and an optional per-block correction factor.
    template <typename ResultT, typename ResultKernel = DefaultKernel<ResultT>, typename ImageWeightFn, typename BlockWeightFn>
    BlockSpMat<ResultT, ResultKernel>* accumulate_weighted_images(
        DistGraph* reference_graph,
        ImageWeightFn&& image_weight_fn,
        BlockWeightFn&& block_weight_fn
    ) {
        static_assert(std::is_constructible<ResultT, T>::value,
                      "ResultT must be constructible from the ImageContainer value type.");

        if (reference_graph == nullptr) {
            throw std::runtime_error("Reference graph must not be null.");
        }

        BlockSpMat<ResultT, ResultKernel>* result = new BlockSpMat<ResultT, ResultKernel>(reference_graph);
        const int n_owned = static_cast<int>(reference_graph->owned_global_indices.size());

        for (const auto& entry : image_blocks) {
            const std::vector<int>& R_vec = entry.first;
            auto* mat_r = entry.second;
            auto* graph_r = mat_r->graph;

            if (graph_r->owned_global_indices != reference_graph->owned_global_indices) {
                throw std::runtime_error("Reference graph must share the same owned rows as the image graphs.");
            }

            const ResultT image_weight = static_cast<ResultT>(image_weight_fn(R_vec));
            if (std::abs(image_weight) <= 1e-12) {
                continue;
            }

            const int n_col_r = static_cast<int>(graph_r->block_sizes.size());
            std::vector<int> col_map(n_col_r, -1);
            for (int local_col = 0; local_col < n_col_r; ++local_col) {
                const int gid = graph_r->get_global_index(local_col);
                auto it = reference_graph->global_to_local.find(gid);
                if (it == reference_graph->global_to_local.end()) {
                    throw std::runtime_error("Image graph column not found in reference graph.");
                }
                col_map[local_col] = it->second;
            }

            #pragma omp parallel for
            for (int local_row = 0; local_row < n_owned; ++local_row) {
                const int start = mat_r->row_ptr[local_row];
                const int end = mat_r->row_ptr[local_row + 1];
                const int row_dim = graph_r->block_sizes[local_row];

                for (int k = start; k < end; ++k) {
                    const int local_col_r = mat_r->col_ind[k];
                    const int local_col_ref = col_map[local_col_r];
                    const ResultT block_weight =
                        image_weight * static_cast<ResultT>(block_weight_fn(local_row, local_col_ref));

                    if (std::abs(block_weight) <= 1e-12) {
                        continue;
                    }

                    const int col_dim = graph_r->block_sizes[local_col_r];
                    const int n_elem = static_cast<int>(mat_r->block_size_elements(k));
                    const T* block_data = mat_r->block_data(k);
                    std::vector<ResultT> block_res(static_cast<size_t>(n_elem));

                    for (int idx = 0; idx < n_elem; ++idx) {
                        block_res[static_cast<size_t>(idx)] =
                            static_cast<ResultT>(block_data[idx]) * block_weight;
                    }

                    result->update_local_block(
                        local_row,
                        local_col_ref,
                        block_res.data(),
                        row_dim,
                        col_dim,
                        AssemblyMode::ADD,
                        MatrixLayout::ColMajor);
                }
            }
        }

        return result;
    }

    template <typename ResultT, typename ResultKernel = DefaultKernel<ResultT>, typename ImageWeightFn>
    BlockSpMat<ResultT, ResultKernel>* accumulate_weighted_images(
        DistGraph* reference_graph,
        ImageWeightFn&& image_weight_fn
    ) {
        return accumulate_weighted_images<ResultT, ResultKernel>(
            reference_graph,
            std::forward<ImageWeightFn>(image_weight_fn),
            [](int, int) { return ResultT(1.0); });
    }

    // Sample K
    // Returns a new BlockSpMat allocated on the base_graph
    BlockSpMat<ResultT, ResultKernel>* sample_k(const std::vector<double>& K, PhaseConvention convention) {
        auto image_weight = [&](const std::vector<int>& R_vec) -> ResultT {
            const double phase_r = -2.0 * M_PI * (K[0] * R_vec[0] + K[1] * R_vec[1] + K[2] * R_vec[2]);
            return std::exp(std::complex<double>(0.0, phase_r));
        };

        if (convention == PhaseConvention::R_ONLY) {
            return accumulate_weighted_images<ResultT, ResultKernel>(base_graph, image_weight);
        }

        return accumulate_weighted_images<ResultT, ResultKernel>(
            base_graph,
            image_weight,
            [&](int local_row, int local_col_base) -> ResultT {
                double ri[3], rj[3];
                atom_data->get_pos(local_row, &ri[0], &ri[1], &ri[2]);
                atom_data->get_pos(local_col_base, &rj[0], &rj[1], &rj[2]);

                double dx = rj[0] - ri[0];
                double dy = rj[1] - ri[1];
                double dz = rj[2] - ri[2];
                atom_data->invert_cell(&dx, &dy, &dz);

                const double phase_pos = -2.0 * M_PI * (K[0] * dx + K[1] * dy + K[2] * dz);
                return std::exp(std::complex<double>(0.0, phase_pos));
            });
    }
};

} // namespace atomic
} // namespace vbcsr
