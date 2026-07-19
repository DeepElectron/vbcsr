#pragma once
#include "atomic_data.hpp"
#include "../block_csr.hpp"
#include <map>
#include <unordered_map>
#include <cstring>
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

template <typename T>
class ImageContainer {
public:
    AtomicData* atom_data;
    DistGraph* base_graph; // Reference to atom_data->graph
    
    // Map R vector (rx, ry, rz) to its specific graph and matrix
    std::map<std::vector<int>, DistGraph*> image_graphs;
    std::map<std::vector<int>, BlockSpMat<T>*> image_blocks;

    // sample_k always yields complex blocks (phase factors), even for real T,
    // so the k-sampled result type is fixed to complex<double>.
    using ResultT = std::complex<double>;

    // True when this container owns ``base_graph`` (the union-graph ctor below);
    // the normal ctor borrows ``data->graph`` and leaves this false.
    bool owns_base_graph = false;
    // True when this container was built with the owning ``ImageContainer(AtomicData*, true)``
    // ctor and must delete ``atom_data`` (and, through it, its graph) in the destructor.
    bool owns_atom_data = false;

public:
    // ``own_atom_data`` lets the container take ownership of a freshly-built AtomicData (e.g.
    // ``get_atomicdata3b``, doc/design/41) and delete it in the destructor — so a graph3b
    // operator/weight container can be built with this NORMAL ctor (which reads the per-edge
    // shifts to build the per-R image graphs) instead of the union-graph ctor below.
    ImageContainer(AtomicData* data, bool own_atom_data = false)
        : atom_data(data), base_graph(data->graph), owns_atom_data(own_atom_data) {
        build_image_graphs();
    }

    ~ImageContainer() {
        for (auto& kv : image_blocks) {
            delete kv.second;
        }
        for (auto& kv : image_graphs) {
            delete kv.second;
        }
        if (owns_base_graph) {
            delete base_graph;
        }
        if (owns_atom_data) {
            delete atom_data;  // deletes its own_graph too (AtomicData dtor)
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
            
            // Also allocate the matrix. The graph stays owned by image_graphs
            // (both maps are deleted together in the destructor).
            BlockSpMat<T>* mat = new BlockSpMat<T>(g);
            image_blocks[R] = mat;
        }
    }

    void add_block(const std::vector<int>& R, int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = kCanonicalBlockLayout) {
        auto it = image_blocks.find(R);
        if (it == image_blocks.end()) {
             // A new R cannot be materialized here: DistGraph construction is
             // collective, so a single rank's add_block cannot create the image
             // graph. Every R must already appear in the AtomicData edge set
             // (the base graph is the union of all image graphs).
             throw std::runtime_error("R vector not found in ImageContainer (must be present in AtomicData)");
        }
        
        it->second->add_block(global_row, global_col, data, rows, cols, mode, layout);
    }
    
    void assemble() {
        for (auto& kv : image_blocks) {
            kv.second->assemble();
        }
    }

    // In-place per-image axpy: ``this += alpha * other``, matching images by their R shift. Used
    // to fuse the 2-body kinetic T into the graph3b V_nl to form the combined static Hamiltonian
    // H_static = T + V_nl (doc/design/41 §3), so only {S, H_static} is sent down (not 3 images)
    // and V_nl is never rebuilt on the pool graph. ``this`` MUST be a superset graph — every R in
    // ``other`` present here, and (per-R) every (row,col) of ``other`` present in this image — the
    // graph3b ⊇ 2-body case. Both sides must share the owned-row partition + block sizes; the
    // cross-graph ``BlockSpMat::axpby`` maps blocks by GLOBAL (row,col) index.
    void axpy_into(const ImageContainer& other, T alpha) {
        for (const auto& kv : other.image_blocks) {
            auto it = image_blocks.find(kv.first);
            if (it == image_blocks.end()) {
                throw std::runtime_error(
                    "ImageContainer::axpy_into: a shift R of `other` is absent in this container "
                    "(this must be a superset graph, e.g. graph3b absorbing the 2-body operator)");
            }
            it->second->axpy(alpha, *kv.second);
        }
    }

    // Read one image block (R, global row, global col) on the owner of the row;
    // returns empty if this rank does not hold it. RowMajor by default.
    std::vector<T> get_block(const std::vector<int>& R, int g_row, int g_col,
                             MatrixLayout layout = MatrixLayout::RowMajor) const {
        auto it = image_blocks.find(R);
        if (it == image_blocks.end()) return {};
        const DistGraph* g = it->second->graph;
        auto lr = g->global_to_local.find(g_row);
        auto lc = g->global_to_local.find(g_col);
        if (lr == g->global_to_local.end() || lc == g->global_to_local.end()) return {};
        // global_to_local also resolves GHOST atoms, whose local ids sit past the owned rows
        // and therefore index no CSR row. Only the owner of a row can serve its blocks.
        if (lr->second < 0 || lr->second >= static_cast<int>(g->owned_global_indices.size())) {
            return {};
        }
        return it->second->get_block(lr->second, lc->second, layout);
    }

    // Batched cross-comm redistribute of ALL images (doc/design/35 incr3). Move every
    // image block from this container's partition to ``target``'s partition in a SINGLE
    // Alltoallv on ``common_comm``, filling ``target``'s image_blocks in place. Source
    // and target cover the same global geometry (same R set, same per-R global
    // adjacency) but different atom-row partitions. The atom-row -> target-owner map is
    // shared across every R (every image graph has the same owned atom rows), so it is
    // derived with ONE Allgatherv; all R then ride one Alltoallv keyed by (R, I, J).
    // ``op`` = Copy (each block to every target owner of its row = send-down/broadcast)
    // or Sum (partials from several source ranks accumulate at the target owner =
    // reduce-up). ``target`` must be a fully-built container (graphs assembled) on the
    // same geometry. No per-R assemble: routed blocks land local, so add_block writes
    // them directly (sample_k reads block_data, which is then current).
    void redistribute_into(ImageContainer& target, RedistOp op, MPI_Comm common_comm) const {
        int cc_size;
        MPI_Comm_size(common_comm, &cc_size);

        // 1. atom-row -> owner cc-ranks in the target partition (shared by all R).
        const std::vector<int>& tgt_owned = target.base_graph->owned_global_indices;
        const int my_n = static_cast<int>(tgt_owned.size());
        std::vector<int> all_n(cc_size);
        MPI_Allgather(&my_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, common_comm);
        std::vector<int> displ(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) displ[i + 1] = displ[i] + all_n[i];
        std::vector<int> all_owned(displ[cc_size]);
        MPI_Allgatherv(tgt_owned.data(), my_n, MPI_INT,
                       all_owned.data(), all_n.data(), displ.data(), MPI_INT, common_comm);
        std::unordered_map<int, std::vector<int>> target_owners;
        for (int r = 0; r < cc_size; ++r) {
            for (int k = displ[r]; k < displ[r + 1]; ++k) {
                target_owners[all_owned[k]].push_back(r);
            }
        }

        // 2. Pack every owned block of every image, keyed by (R, I, J), once per owner.
        //    Header per block: rx, ry, rz, gi, gj, r_dim, c_dim (7 ints) + block
        //    data in canonical row-major layout (kCanonicalBlockLayout), copied
        //    straight from backend storage and unpacked with the same constant.
        const int header_ints = 7;
        std::vector<size_t> send_counts(cc_size, 0);
        auto for_each_owned_block = [&](auto&& fn) {
            for (const auto& kv : image_blocks) {
                const std::vector<int>& R = kv.first;
                const BlockSpMat<T>* mat = kv.second;
                const DistGraph* g = mat->graph;
                const int n_owned = static_cast<int>(g->owned_global_indices.size());
                for (int i = 0; i < n_owned; ++i) {
                    const int gi = g->owned_global_indices[i];
                    auto it = target_owners.find(gi);
                    if (it == target_owners.end()) continue;
                    const int r_dim = g->block_sizes[i];
                    for (int k = g->adj_ptr[i]; k < g->adj_ptr[i + 1]; ++k) {
                        const int lcol = g->adj_ind[k];
                        const int gj = g->get_global_index(lcol);
                        const int c_dim = g->block_sizes[lcol];
                        fn(R, gi, gj, r_dim, c_dim, mat->block_data(k), it->second);
                    }
                }
            }
        };
        for_each_owned_block([&](const std::vector<int>&, int, int, int r_dim, int c_dim,
                                 const T*, const std::vector<int>& owners) {
            const size_t blk = header_ints * sizeof(int)
                             + static_cast<size_t>(r_dim) * c_dim * sizeof(T);
            for (int dest : owners) send_counts[dest] += blk;
        });
        std::vector<size_t> sdispls(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) sdispls[i + 1] = sdispls[i] + send_counts[i];
        std::vector<char> send_blob(sdispls[cc_size]);
        std::vector<size_t> off = sdispls;
        for_each_owned_block([&](const std::vector<int>& R, int gi, int gj, int r_dim,
                                 int c_dim, const T* data, const std::vector<int>& owners) {
            const size_t data_bytes = static_cast<size_t>(r_dim) * c_dim * sizeof(T);
            const int hdr[header_ints] = {R[0], R[1], R[2], gi, gj, r_dim, c_dim};
            for (int dest : owners) {
                char* ptr = send_blob.data() + off[dest];
                std::memcpy(ptr, hdr, sizeof(hdr)); ptr += sizeof(hdr);
                std::memcpy(ptr, data, data_bytes); ptr += data_bytes;
                off[dest] = ptr - send_blob.data();
            }
        });

        // 3. One Alltoallv on the common comm.
        std::vector<size_t> recv_counts(cc_size);
        MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE,
                     recv_counts.data(), sizeof(size_t), MPI_BYTE, common_comm);
        std::vector<size_t> rdispls(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) rdispls[i + 1] = rdispls[i] + recv_counts[i];
        std::vector<char> recv_blob(rdispls[cc_size]);
        safe_alltoallv(send_blob.data(), send_counts, sdispls, MPI_BYTE,
                       recv_blob.data(), recv_counts, rdispls, MPI_BYTE, common_comm);

        // 4. Place received blocks on the target images (zero-then-ADD for Sum).
        const AssemblyMode amode = (op == RedistOp::Sum) ? AssemblyMode::ADD
                                                         : AssemblyMode::INSERT;
        if (op == RedistOp::Sum) {
            for (auto& kv : target.image_blocks) kv.second->fill(T(0));
        }
        const char* ptr = recv_blob.data();
        const char* rend = recv_blob.data() + recv_blob.size();
        while (ptr < rend) {
            int hdr[header_ints];
            std::memcpy(hdr, ptr, sizeof(hdr)); ptr += sizeof(hdr);
            const std::vector<int> R = {hdr[0], hdr[1], hdr[2]};
            const int gi = hdr[3], gj = hdr[4], r_dim = hdr[5], c_dim = hdr[6];
            auto it = target.image_blocks.find(R);
            if (it == target.image_blocks.end()) {
                throw std::runtime_error("ImageContainer::redistribute_into: received an R "
                                         "absent from the target (geometry mismatch).");
            }
            it->second->add_block(gi, gj, reinterpret_cast<const T*>(ptr), r_dim, c_dim,
                                  amode, kCanonicalBlockLayout);
            ptr += static_cast<size_t>(r_dim) * c_dim * sizeof(T);
        }
    }

    // NOTE: the former ``gather_into`` (request-list gather that re-``assemble``d into a target
    // DistGraph) was removed in doc/design/41 §2.3 — its ``assemble`` re-routes the just-fetched
    // blocks by the target partition (a no-op only for a serial target, and an un-gather for any
    // real partition). The operator-walk gather is now the keyed ``LocalBlockView`` produced by
    // ``rescupp::lcao_grid::GatherPlan::gather_local`` (alias local, cache remote, no re-route).

    // Accumulate all image blocks onto a compatible reference graph using
    // a per-image weight and an optional per-block correction factor.
    template <typename ResultT, typename ImageWeightFn, typename BlockWeightFn>
    BlockSpMat<ResultT>* accumulate_weighted_images(
        DistGraph* reference_graph,
        ImageWeightFn&& image_weight_fn,
        BlockWeightFn&& block_weight_fn
    ) {
        static_assert(std::is_constructible<ResultT, T>::value,
                      "ResultT must be constructible from the ImageContainer value type.");

        if (reference_graph == nullptr) {
            throw std::runtime_error("Reference graph must not be null.");
        }

        BlockSpMat<ResultT>* result = new BlockSpMat<ResultT>(reference_graph);
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
                const int start = mat_r->row_ptr()[local_row];
                const int end = mat_r->row_ptr()[local_row + 1];
                const int row_dim = graph_r->block_sizes[local_row];

                for (int k = start; k < end; ++k) {
                    const int local_col_r = mat_r->col_ind()[k];
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
                        kCanonicalBlockLayout);
                }
            }
        }

        return result;
    }

    template <typename ResultT, typename ImageWeightFn>
    BlockSpMat<ResultT>* accumulate_weighted_images(
        DistGraph* reference_graph,
        ImageWeightFn&& image_weight_fn
    ) {
        return accumulate_weighted_images<ResultT>(
            reference_graph,
            std::forward<ImageWeightFn>(image_weight_fn),
            [](int, int) { return ResultT(1.0); });
    }

    // Sample K
    // Returns a new BlockSpMat allocated on the base_graph
    BlockSpMat<ResultT>* sample_k(const std::vector<double>& K, PhaseConvention convention) {
        auto image_weight = [&](const std::vector<int>& R_vec) -> ResultT {
            const double phase_r = -2.0 * M_PI * (K[0] * R_vec[0] + K[1] * R_vec[1] + K[2] * R_vec[2]);
            return std::exp(std::complex<double>(0.0, phase_r));
        };

        if (convention == PhaseConvention::R_ONLY) {
            return accumulate_weighted_images<ResultT>(base_graph, image_weight);
        }

        return accumulate_weighted_images<ResultT>(
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
