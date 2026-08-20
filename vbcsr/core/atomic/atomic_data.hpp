#pragma once
#include "../dist_graph.hpp"
#include <vector>
#include <map>
#include <set>
#include <array>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>
#include <cstdint>
#include "neighbourlist.hpp"
#include "io.hpp"
#include "distributed_build.hpp"
#include <memory>

#ifdef VBCSR_HAVE_PARMETIS
#include <parmetis.h>
#endif

namespace vbcsr {
namespace atomic {

/// Inverse of a row-major 3x3 cell, so a Cartesian offset can be read in
/// fractional coordinates.
///
/// A free function rather than a method because callers that transform many
/// vectors -- grid collocation, minimum-image searches -- want the matrix once
/// and reuse it, where AtomicData::invert_cell recomputes it per vector.
inline std::array<double, 9> InvertCell3x3(const double* cell) {
    const double a = cell[0], b = cell[1], c = cell[2];
    const double d = cell[3], e = cell[4], f = cell[5];
    const double g = cell[6], h = cell[7], i = cell[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!(std::abs(det) > 1e-12)) {
        throw std::runtime_error("Cell is singular; it has no fractional coordinates.");
    }
    const double s = 1.0 / det;
    return {(e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s,
            (f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s,
            (d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s};
}

class AtomicData {
public:
    DistGraph* graph = nullptr;
    bool own_graph = false;

    // Atom attributes (Local + Ghost)
    // Indexed by local index from DistGraph (0 to n_owned+n_ghost-1)
    std::vector<int> atom_type={};
    std::vector<int> atomic_numbers={};
    std::vector<int> atom_index={}; // Original ID from file
    std::vector<double> x={}; std::vector<double> y={}; std::vector<double> z={};
    
    // Global info
    std::vector<int> type_norb={};
    std::vector<double> cell={}; // 9 elements
    std::vector<bool> pbc={false, false, false};
    
    // Edge storage
    // We store edges as provided in input, mapped to local indices.
    struct Edge {
        int src; // Local index of source (must be owned)
        int dst; // Local index of dest (owned or ghost)
        int rx, ry, rz;
    };
    std::vector<Edge> edges={};
    
    // Connectivity: local atom index -> list of edge indices in 'edges' vector
    std::vector<std::vector<int>> iconn={};

    // Offset for global indexing
    int atom_offset=0;
    int n_atom=0;
    int N_atom=0;
    int n_edge=0;
    int N_edge=0;

    // Provenance of a Structure-expanded graph. File-backed real-space
    // operators use it to map supercell atom/image keys back to the primitive
    // (i,j,R) convention. Ordinary and caller-partitioned graphs keep the
    // identity defaults.
    int primitive_n_atom=0;
    std::array<int, 3> structure_supercell{{1, 1, 1}};
    
    // Offsets for block-sparse matrix
    std::vector<int> local_offsets={}; // Local offset of each atom's block
    std::vector<long long> global_offsets={}; // Global offset of each atom's block
    
    MPI_Comm comm=MPI_COMM_WORLD;
    int rank=0, size=1;

    AtomicData(DistGraph* g) : graph(g), own_graph(false) {
        comm = graph->comm;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }
        compute_offsets();
    }

    AtomicData(MPI_Comm comm_) : comm(comm_), own_graph(false) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }
    }

    AtomicData(
      size_t n_atom_, size_t N_atom_, size_t atom_offset_, size_t n_edge_, size_t N_edge_,
      const int *atom_index_in, const int *atom_type_in, const int *edge_index_in, const int *type_norb_in, const int *edge_shift_vec_in,
      const double *cell_in, const double *pos_in,
      MPI_Comm comm_,
      const std::vector<bool>& pbc_in = std::vector<bool>{}
    ) : AtomicData(
            n_atom_,
            N_atom_,
            atom_offset_,
            n_edge_,
            N_edge_,
            atom_index_in,
            atom_type_in,
            edge_index_in,
            type_norb_in,
            edge_shift_vec_in,
            cell_in,
            pos_in,
            nullptr,
            comm_,
            pbc_in) {}

    AtomicData(
      size_t n_atom_, size_t N_atom_, size_t atom_offset_, size_t n_edge_, size_t N_edge_,
      const int *atom_index_in, const int *atom_type_in, const int *edge_index_in, const int *type_norb_in, const int *edge_shift_vec_in,
      const double *cell_in, const double *pos_in, const int *atomic_numbers_in,
      MPI_Comm comm_,
      const std::vector<bool>& pbc_in = std::vector<bool>{}
    ) : atom_offset(atom_offset_), n_atom(n_atom_), N_atom(N_atom_), n_edge(n_edge_), N_edge(N_edge_), comm(comm_), own_graph(true) {
        
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }

        pbc = pbc_in.empty() ? infer_pbc_from_edge_shifts(n_edge, edge_shift_vec_in, initialized, comm)
                             : pbc_in;

        if (pbc.size() != 3) {
            throw std::runtime_error("pbc must have exactly 3 elements");
        }

        // 1. Setup Graph
        // Prepare adjacency for DistGraph (remove duplicates/R)
        std::vector<int> owned_indices(n_atom);
        std::vector<int> my_block_sizes(n_atom);
        std::vector<std::vector<int>> adj(n_atom);
        
        // Determine max_type and populate type_norb
        int max_type = 0;
        for(size_t i=0; i<n_atom; ++i) max_type = std::max(max_type, atom_type_in[i]);
        int my_max_type = max_type;
        int global_max_type;
        if (initialized) {
            MPI_Allreduce(&my_max_type, &global_max_type, 1, MPI_INT, MPI_MAX, comm);
        } else {
            global_max_type = my_max_type;
        }
        type_norb.assign(type_norb_in, type_norb_in + global_max_type + 1);

        for(int i=0; i<n_atom; ++i) {
            owned_indices[i] = i + atom_offset;
            my_block_sizes[i] = type_norb[atom_type_in[i]];
        }

        // Build adj
        for(size_t k=0; k<n_edge; ++k) {
            int src_gid = edge_index_in[2*k];
            int dst_gid = edge_index_in[2*k+1];
            
            // src should be local
            if (src_gid < atom_offset || src_gid >= atom_offset + n_atom) {
                 throw std::runtime_error("Edge source not local");
            }
            int src_lid = src_gid - atom_offset;
            adj[src_lid].push_back(dst_gid);
        }

        // add onsite edges
        for(int i=0; i<n_atom; ++i) {
            adj[i].push_back(i + atom_offset);
        }
        
        // Remove duplicates
        for(auto& neighbors : adj) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        
        graph = new DistGraph(comm); 
        graph->construct_distributed(owned_indices, my_block_sizes, adj);
        
        compute_offsets();
        
        // 2. Store Local Data
        int n_owned = n_atom; 
        int n_ghost = graph->ghost_global_indices.size();
        int total_local = n_owned + n_ghost;
        
        atom_type.resize(total_local);
        atomic_numbers.resize(total_local, -1);
        atom_index.resize(total_local);
        x.resize(total_local);
        y.resize(total_local);
        z.resize(total_local);
        
        for(int i=0; i<n_owned; ++i) {
            atom_type[i] = atom_type_in[i];
            atomic_numbers[i] = atomic_numbers_in != nullptr ? atomic_numbers_in[i] : -1;
            atom_index[i] = atom_index_in[i];
            x[i] = pos_in[3*i];
            y[i] = pos_in[3*i+1];
            z[i] = pos_in[3*i+2];
        }
        
        cell.assign(cell_in, cell_in + 9);
        
        // 3. Fetch Ghost Data
        exchange_attribute(atom_type);
        exchange_attribute(atomic_numbers);
        exchange_attribute(atom_index);
        exchange_attribute(x);
        exchange_attribute(y);
        exchange_attribute(z);
        
        // 4. Store Edges (mapped to local indices)
        edges.resize(n_edge);
        iconn.resize(n_owned);
        
        for(size_t k=0; k<n_edge; ++k) {
            int src_gid = edge_index_in[2*k];
            int dst_gid = edge_index_in[2*k+1];
            
            int src_lid = graph->global_to_local.at(src_gid);
            int dst_lid = graph->global_to_local.at(dst_gid); 
            
            edges[k] = {src_lid, dst_lid, edge_shift_vec_in[3*k], edge_shift_vec_in[3*k+1], edge_shift_vec_in[3*k+2]};
            
            iconn[src_lid].push_back(k);
        }
    }
    
    static AtomicData* from_points(const std::vector<double>& pos, const std::vector<int>& z, 
                                   const std::vector<double>& cell, const std::vector<bool>& pbc, 
                                   const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb_in, 
                                   MPI_Comm comm) {
        int rank, size;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }

        // 1. Spread the atoms and build the edges, both distributed.
        //
        // Rank 0 holds the geometry on the way in and nothing else: the atoms
        // are scattered in blocks, repartitioned by inertial bisection, and each
        // rank then runs a neighbour search over its own atoms plus a halo. The
        // alternative -- one neighbour list over every atom on rank 0 -- costs
        // that rank N * <neighbours>, which at 1e6 atoms is gigabytes and tens
        // of seconds before any other rank has done anything.
        std::vector<int> type_norb = type_norb_in;
        std::vector<double> my_pos;
        std::vector<int> my_z;
        std::vector<int> my_types;
        std::vector<int> my_indices;
        std::vector<int> my_edges_flat;
        int my_n_atom = 0;
        int n_global = 0;

        distribute_and_build_edges(comm, rank, size, pos, z, cell, pbc, r_max_per_type,
                                   type_norb, n_global, my_n_atom, my_pos, my_z, my_types,
                                   my_indices, my_edges_flat);

        if (n_global == 0) {
            return new AtomicData(comm);
        }

        // 2. ParMETIS Partitioning
        //
        // Bisection already put neighbours on the same rank; this refines that
        // for edge cut and balance, and the redistribution below is the same one
        // it always was.
        std::vector<int> vtxdist;
        std::vector<int> xadj;
        std::vector<int> adjncy;
        int my_start;
        
        // this step build temporary graph for partitioning
        build_parmetis_graph(comm, rank, size, my_n_atom, my_edges_flat, vtxdist, xadj, adjncy, my_start);
        
        std::vector<int> part(my_n_atom);
        std::fill(part.begin(), part.end(), rank);

        // Balance orbitals, not atoms -- the same weight the inertial
        // bisection above used, so the refinement improves the edge cut of
        // that partition instead of rebalancing it against a different
        // objective.
        std::vector<int> my_vwgt(my_n_atom);
        for (int i = 0; i < my_n_atom; ++i) my_vwgt[i] = type_norb[my_types[i]];

        partition_graph(vtxdist, xadj, adjncy, size, part, comm, my_pos, n_global, my_vwgt);
        
        // 3. Redistribute Atoms
        std::vector<double> r_pos;
        std::vector<int> r_z;
        std::vector<int> r_types;
        std::vector<int> r_indices;
        int total_recv;
        
        std::vector<int> my_inter_indices(my_n_atom);
        for(int i=0; i<my_n_atom; ++i) my_inter_indices[i] = my_start + i;
        
        std::vector<int> r_inter_indices;
        redistribute_atoms(comm, rank, size, my_n_atom, part, my_pos, my_z, my_types, my_indices, my_inter_indices,
                           r_pos, r_z, r_types, r_indices, r_inter_indices, total_recv);
        
        // 4. Redistribute Edges
        std::vector<int> r_edges;
        redistribute_edges(comm, rank, size, my_start, my_edges_flat, part, r_edges);
        
        // 5. Re-map IDs to be contiguous on each rank
        std::vector<int> all_recv_counts(size);
        if (initialized) {
            MPI_Allgather(&total_recv, 1, MPI_INT, all_recv_counts.data(), 1, MPI_INT, comm);
        } else {
            all_recv_counts[0] = total_recv;
        }
        std::vector<int> all_recv_displs(size + 1, 0);
        for(int i=0; i<size; ++i) all_recv_displs[i+1] = all_recv_displs[i] + all_recv_counts[i];
        
        std::vector<int> all_r_inter_indices(n_global);
        if (initialized) {
            MPI_Allgatherv(r_inter_indices.data(), total_recv, MPI_INT, all_r_inter_indices.data(), all_recv_counts.data(), all_recv_displs.data(), MPI_INT, comm);
        } else {
            std::copy(r_inter_indices.begin(), r_inter_indices.end(), all_r_inter_indices.begin());
        }
        
        std::vector<int> inter_to_final(n_global);
        for(int i=0; i<n_global; ++i) {
            inter_to_final[all_r_inter_indices[i]] = i;
        }
        
        // Update r_edges to use final IDs
        for(size_t k=0; k < r_edges.size() / 5; ++k) {
            r_edges[5*k] = inter_to_final[r_edges[5*k]];
            r_edges[5*k+1] = inter_to_final[r_edges[5*k+1]];
        }
                      
        // 6. Construct AtomicData
        return construct_final_object(
            comm,
            rank,
            size,
            cell,
            pbc,
            total_recv,
            r_indices,
            r_z,
            r_types,
            r_pos,
            r_edges,
            type_norb);
    }

    // Distributed construction from a CALLER-GIVEN partition (doc/design/42 §4).
    // Each rank passes ONLY its owned atoms — there is no rank-0 gather of the
    // geometry and no ParMETIS: the partition is whatever the caller chose (e.g.
    // a spatial slab/RCB aligned with the grid). Global ids are assigned
    // contiguously by rank (Allgather of counts -> Exscan), so rank r owns
    // [offset_r, offset_r + n_r). This form Allgathers positions/Z (O(N), transient)
    // so every rank builds the edges of ITS OWNED atoms locally; the persistent
    // storage is owned + ghost only. ``my_input_index[li]`` is the original
    // input-order index of owned atom ``li`` (fills AtomicData.atom_index).
    static AtomicData* from_distributed(
        const std::vector<double>& my_pos, const std::vector<int>& my_z,
        const std::vector<int>& my_input_index,
        const std::vector<double>& cell, const std::vector<bool>& pbc,
        const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb_in,
        MPI_Comm comm) {
        int rank = 0, size = 1, initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) { MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size); }
        const int my_n = static_cast<int>(my_z.size());

        // 1. Contiguous global-id layout: rank r owns [offset_r, offset_r + n_r).
        std::vector<int> counts(size, my_n), displs(size + 1, 0);
        if (initialized) MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        for (int i = 0; i < size; ++i) displs[i + 1] = displs[i] + counts[i];
        const int total = displs[size];
        const int my_offset = displs[rank];
        if (total == 0) return new AtomicData(comm);

        // 2. Allgather all positions + atomic numbers (global-id == rank order).
        std::vector<int> pos_counts(size), pos_displs(size + 1, 0);
        for (int i = 0; i < size; ++i) { pos_counts[i] = counts[i] * 3; pos_displs[i + 1] = pos_displs[i] + pos_counts[i]; }
        std::vector<double> all_pos(static_cast<size_t>(total) * 3);
        std::vector<int> all_z(total);
        if (initialized) {
            MPI_Allgatherv(my_pos.data(), my_n * 3, MPI_DOUBLE, all_pos.data(), pos_counts.data(), pos_displs.data(), MPI_DOUBLE, comm);
            MPI_Allgatherv(my_z.data(), my_n, MPI_INT, all_z.data(), counts.data(), displs.data(), MPI_INT, comm);
        } else { all_pos = my_pos; all_z = my_z; }

        // 3. Global z -> type (sorted-unique z; identical on every rank).
        std::vector<int> uz = all_z; std::sort(uz.begin(), uz.end());
        uz.erase(std::unique(uz.begin(), uz.end()), uz.end());
        std::map<int, int> z2t; for (size_t i = 0; i < uz.size(); ++i) z2t[uz[i]] = static_cast<int>(i);
        std::vector<int> all_types(total); for (int g = 0; g < total; ++g) all_types[g] = z2t[all_z[g]];
        std::vector<int> type_norb = type_norb_in;
        if (type_norb.empty()) type_norb.assign(uz.size(), 1);
        else if (type_norb.size() < uz.size()) type_norb.resize(uz.size(), type_norb.back());
        // Same undersized-cutoff padding as process_input_rank0.
        std::vector<double> r_max_type = r_max_per_type;
        if (r_max_type.empty()) r_max_type.assign(uz.size(), 0.0);
        else if (r_max_type.size() < uz.size()) r_max_type.resize(uz.size(), r_max_type.back());
        std::vector<int> my_types(my_n); for (int li = 0; li < my_n; ++li) my_types[li] = all_types[my_offset + li];

        // 4. Edges of MY owned atoms (NeighborList over all positions; same cutoff
        //    + distance test as process_input_rank0, so the graph is identical to
        //    from_points for the same geometry). r_edges: 5 ints {gi, gj, rx, ry, rz}.
        double max_r = 0; for (double r : r_max_type) max_r = std::max(max_r, r);
        NeighborList nl; nl.build(all_pos, cell, pbc, max_r * 2.0);
        std::vector<int> r_edges;
        for (int li = 0; li < my_n; ++li) {
            const int gi = my_offset + li, ti = all_types[gi];
            for (const auto& nb : nl.get_neighbors(gi)) {
                const int gj = nb.index, tj = all_types[gj];
                const double rc = r_max_type[ti] + r_max_type[tj];
                const double dx = all_pos[3 * gj]     - all_pos[3 * gi]     + nb.rx * cell[0] + nb.ry * cell[3] + nb.rz * cell[6];
                const double dy = all_pos[3 * gj + 1] - all_pos[3 * gi + 1] + nb.rx * cell[1] + nb.ry * cell[4] + nb.rz * cell[7];
                const double dz = all_pos[3 * gj + 2] - all_pos[3 * gi + 2] + nb.rx * cell[2] + nb.ry * cell[5] + nb.rz * cell[8];
                if (std::sqrt(dx * dx + dy * dy + dz * dz) > rc + 1e-9) continue;
                r_edges.push_back(gi); r_edges.push_back(gj);
                r_edges.push_back(nb.rx); r_edges.push_back(nb.ry); r_edges.push_back(nb.rz);
            }
        }

        // 5. Build the distributed object (owned atoms + owned-src edges; the ctor
        //    derives ghosts from the edge dsts). Same final assembly as from_points.
        return construct_final_object(comm, rank, size, cell, pbc, my_n,
            my_input_index, my_z, my_types, my_pos, r_edges, type_norb);
    }

    /// Reads a structure file on rank 0 and builds its graph.
    ///
    /// Only the parse is serial; from_points spreads the atoms before it does
    /// any geometry. Per-type tables are indexed by the distinct atomic numbers
    /// in ascending order -- ReadStructure(...).TypeSymbols() is that list, for
    /// a caller that needs to see it before choosing them.
    static AtomicData* from_file(const std::string& filename,
                                 const std::vector<double>& r_max_per_type,
                                 const std::vector<int>& type_norb,
                                 MPI_Comm comm, const std::string& format = "") {
        int rank = 0;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Comm_rank(comm, &rank);

        Structure data;
        if (rank == 0) data = ReadStructure(filename, format);

        return from_points(data.pos, data.z, data.cell, data.pbc, r_max_per_type,
                           type_norb, comm);
    }

    ~AtomicData() {
        if (own_graph && graph) delete graph;
    }

    // Accessors
    DistGraph* get_graph() { return graph; }
    const double* cell_ptr() const { return cell.data(); }
    int n_type() const { return type_norb.size(); } 
    
    void get_vertex_range_local(int *start, int *end) const {
        *start = 0; *end = n_atom + graph->ghost_global_indices.size();
    }

    void get_edge_range_local(int *start, int *end) const {
        *start = 0; *end = n_edge;
    }
    
    bool atom_is_ghost(int idx) const {
        return idx >= n_atom;
    }
    
    void get_atom_norb(int idx, int *norb) const {
        *norb = type_norb[atom_type[idx]];
    }
    
    void get_atom_id(int idx, int *aid) const {
        *aid = atom_index[idx];
    }
    
    void get_atom_type(int idx, int *tp) const {
        *tp = atom_type[idx];
    }

    bool has_atomic_numbers() const {
        if (atomic_numbers.size() < static_cast<size_t>(n_atom)) {
            return false;
        }
        for (int i = 0; i < n_atom; ++i) {
            if (atomic_numbers[i] < 0) {
                return false;
            }
        }
        return true;
    }

    void ensure_owned_atomic_numbers(const char* context) const {
        if (!has_atomic_numbers()) {
            throw std::runtime_error(
                std::string(context) +
                " requires explicit atomic numbers. "
                "Pass atomic_numbers=... when using AtomicData::from_distributed.");
        }
    }
    
    void get_pos(int idx, double *rx, double *ry, double *rz) const {
        *rx = x[idx];
        *ry = y[idx];
        *rz = z[idx];
    }
    
    int get_global_index(int lid) const {
        if (lid < n_atom) return graph->owned_global_indices[lid];
        return graph->ghost_global_indices[lid - n_atom];
    }
    
    void get_connected_atoms(int edge_idx, int *i, int *j) const {
        *i = edges[edge_idx].src;
        *j = edges[edge_idx].dst;
    }
    
    void get_edge_shift_vec(int edge_idx, int *Rx, int *Ry, int *Rz) const {
        *Rx = edges[edge_idx].rx;
        *Ry = edges[edge_idx].ry;
        *Rz = edges[edge_idx].rz;
    }
    
    void get_offset_local(int idx, int *offset) const {
        *offset = local_offsets[idx];
    }
    
    void get_offset_global(int idx, int *offset) const {
        *offset = static_cast<int>(global_offsets[idx]);
    }

    int get_edge_dst(int edge_idx) const {
        return edges[edge_idx].dst;
    }

    const std::vector<int>& get_atom_edges(int atom_idx) const {
        return iconn[atom_idx];
    }

    void compute_offsets() {
        if (!graph) return;
        int n_local = graph->block_sizes.size();
        local_offsets.resize(n_local);
        global_offsets.resize(n_local);

        // 1. Local offsets (prefix sum of block sizes)
        int current_offset = 0;
        for(int i=0; i<n_local; ++i) {
            local_offsets[i] = current_offset;
            current_offset += graph->block_sizes[i];
        }

        // 2. Global offsets
        int n_owned = graph->owned_global_indices.size();
        long long my_owned_elements = 0;
        for(int i=0; i<n_owned; ++i) {
            my_owned_elements += graph->block_sizes[i];
        }

        long long rank_global_start = 0;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Exscan(&my_owned_elements, &rank_global_start, 1, MPI_LONG_LONG, MPI_SUM, comm);
        }
        if (rank == 0) rank_global_start = 0;

        // Fill owned
        for(int i=0; i<n_owned; ++i) {
            global_offsets[i] = rank_global_start + local_offsets[i];
        }

        // Fill ghosts (Communicate)
        std::vector<long long> send_buf(graph->send_indices.size());
        for(size_t i=0; i<graph->send_indices.size(); ++i) {
            int lid = graph->send_indices[i];
            send_buf[i] = global_offsets[lid];
        }

        std::vector<long long> recv_buf(graph->recv_indices.size());
        
        if (initialized) {
            MPI_Alltoallv(send_buf.data(), graph->send_counts.data(), graph->send_displs.data(), MPI_LONG_LONG,
                        recv_buf.data(), graph->recv_counts.data(), graph->recv_displs.data(), MPI_LONG_LONG, comm);
        }

        for(size_t i=0; i<graph->recv_indices.size(); ++i) {
            int lid = graph->recv_indices[i];
            global_offsets[lid] = recv_buf[i];
        }
    }

    void get_edge_vec(int edge_idx, double *rx, double *ry, double *rz) {
        int i = edges[edge_idx].src;
        int j = edges[edge_idx].dst;
        int Rx = edges[edge_idx].rx;
        int Ry = edges[edge_idx].ry;
        int Rz = edges[edge_idx].rz;
        
        *rx = Rx*cell[0] + Ry*cell[3] + Rz*cell[6];
        *ry = Rx*cell[1] + Ry*cell[4] + Rz*cell[7];
        *rz = Rx*cell[2] + Ry*cell[5] + Rz*cell[8];
        
        *rx += x[j] - x[i];
        *ry += y[j] - y[i];
        *rz += z[j] - z[i];
    }

    // Rewrites a Cartesian vector in fractional coordinates. A degenerate cell
    // is left alone rather than raised on, which is what the callers here
    // expect: a cell-less (molecular) system has nothing to reduce.
    void invert_cell(double *x, double *y, double *z) {
        std::array<double, 9> inv;
        try {
            inv = InvertCell3x3(cell.data());
        } catch (const std::runtime_error&) {
            return;
        }

        const double a = inv[0]*(*x) + inv[1]*(*y) + inv[2]*(*z);
        const double b = inv[3]*(*x) + inv[4]*(*y) + inv[5]*(*z);
        const double c = inv[6]*(*x) + inv[7]*(*y) + inv[8]*(*z);

        *x = a;
        *y = b;
        *z = c;
    }

    // Compute global number of orbitals
    int norb() {
        int local_norb = 0;
        for(int i=0; i<n_atom; ++i) {
            local_norb += type_norb[atom_type[i]];
        }
        int global_norb = 0;
        if (size > 1) {
            MPI_Allreduce(&local_norb, &global_norb, 1, MPI_INT, MPI_SUM, comm);
        } else {
            global_norb = local_norb;
        }
        return global_norb;
    }

    // Compute cell volume
    double volume(std::string axis="abc") {
        // compute cross product of cell vectors
        if (axis == "ab") return std::sqrt(std::pow(cell[1]*cell[5] - cell[2]*cell[4], 2) 
            + std::pow(cell[2]*cell[3] - cell[0]*cell[5], 2) 
            + std::pow(cell[0]*cell[4] - cell[1]*cell[3], 2));
        if (axis == "bc") return std::sqrt(std::pow(cell[4]*cell[8] - cell[5]*cell[7], 2) 
            + std::pow(cell[5]*cell[6] - cell[3]*cell[8], 2) 
            + std::pow(cell[3]*cell[7] - cell[4]*cell[6], 2));
        if (axis == "ca") return std::sqrt(std::pow(cell[7]*cell[2] - cell[8]*cell[1], 2) 
            + std::pow(cell[8]*cell[0] - cell[6]*cell[2], 2) 
            + std::pow(cell[6]*cell[1] - cell[7]*cell[0], 2));
        if (axis == "abc") return std::abs(cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
                   - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
                   + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
        throw std::runtime_error("Invalid axis");
    }

    DistGraph* get_graph3b(const std::vector<double>& r_max_left, const std::vector<double>& r_max_right) {
        // 1. Build reduced connectivity (riconn)
        std::vector<std::vector<int>> riconn(atom_type.size());
        int initialized = 0;
        MPI_Initialized(&initialized);
        
        for(int i=0; i<n_atom; ++i) {
            for(int edge_idx : iconn[i]) {
                int j = edges[edge_idx].dst;
                
                double rx, ry, rz;
                get_edge_vec(edge_idx, &rx, &ry, &rz);
                double r = std::sqrt(rx*rx + ry*ry + rz*rz);
                
                int itype = atom_type[i];
                int jtype = atom_type[j];
                
                if (r <= r_max_left[itype] + r_max_right[jtype] + 1e-9) {
                    riconn[i].push_back(j);
                }
            }
            std::sort(riconn[i].begin(), riconn[i].end());
            riconn[i].erase(std::unique(riconn[i].begin(), riconn[i].end()), riconn[i].end());
        }

        
        std::map<int, std::vector<std::pair<int, int>>> send_map;
        
        for(int i=0; i<n_atom; ++i) {
            const auto& neighbors = riconn[i];
            for(size_t idx_j=0; idx_j<neighbors.size(); ++idx_j) {
                int j = neighbors[idx_j];
                int gid_j = get_gid(j);
                int owner_j = graph->find_owner(gid_j);
                
                for(size_t idx_k=0; idx_k<neighbors.size(); ++idx_k) {
                    int k = neighbors[idx_k];
                    int gid_k = get_gid(k);
                    send_map[owner_j].push_back({gid_j, gid_k});
                }
            }
        }

        for(int i=0; i<size; ++i) {
            std::sort(send_map[i].begin(), send_map[i].end());
            send_map[i].erase(std::unique(send_map[i].begin(), send_map[i].end()), send_map[i].end());
        }
        
        std::vector<int> send_counts(size, 0);
        for(auto& kv : send_map) send_counts[kv.first] = kv.second.size() * 2;
        
        std::vector<int> recv_counts(size);
        if (initialized) {
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        } else {
            std::copy(send_counts.begin(), send_counts.end(), recv_counts.begin());
        }
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<int> send_buf(sdispls[size]);
        for(auto& kv : send_map) {
            int offset = sdispls[kv.first];
            for(const auto& p : kv.second) {
                send_buf[offset++] = p.first;
                send_buf[offset++] = p.second;
            }
        }
        
        std::vector<int> recv_buf(rdispls[size]);
        if (initialized) {
            MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                          recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, comm);
        } else {
            std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
        }
                      
        std::vector<std::vector<int>> matrix_adj(n_atom);
        
        for(size_t i=0; i<recv_buf.size(); i+=2) {
            int gid_j = recv_buf[i];
            int gid_k = recv_buf[i+1];
            
            if (graph->global_to_local.find(gid_j) == graph->global_to_local.end()) {
                 throw std::runtime_error("Received 3-body edge for non-owned atom");
            }
            int lid_j = graph->global_to_local.at(gid_j);
            
            matrix_adj[lid_j].push_back(gid_k);
        }

        // add two-body and onsite edges
        for(int i=0; i<n_atom; ++i) {
            for(int j=graph->adj_ptr[i]; j<graph->adj_ptr[i+1]; ++j) {
                int j_lid = graph->adj_ind[j];
                int j_gid = get_gid(j_lid);
                matrix_adj[i].push_back(j_gid);
            }
        }
        
        for(int i=0; i<n_atom; ++i) {
            std::sort(matrix_adj[i].begin(), matrix_adj[i].end());
            matrix_adj[i].erase(std::unique(matrix_adj[i].begin(), matrix_adj[i].end()), matrix_adj[i].end());
        }
        
        DistGraph* new_graph = new DistGraph(comm);
        
        std::vector<int> owned_indices(n_atom);
        std::iota(owned_indices.begin(), owned_indices.end(), atom_offset);
        
        std::vector<int> my_block_sizes(n_atom);
        for(int i=0; i<n_atom; ++i) {
            my_block_sizes[i] = type_norb[atom_type[i]];
        }
        
        new_graph->construct_distributed(owned_indices, my_block_sizes, matrix_adj);

        return new_graph;
    }

    // Build a full ``AtomicData`` whose neighbour graph IS the 3-centre (graph3b) structure,
    // with the SAME atom distribution as ``this`` — the proper-AtomicData sibling of
    // ``get_graph3b`` (doc/design/41). Unlike ``get_graph3b`` (which returns a shift-less
    // ``DistGraph`` and so forces the union-graph ``ImageContainer`` ctor + a hand-built
    // explicit shift set), this tracks each 3-centre block's integer lattice shift, so the
    // normal ``ImageContainer(AtomicData*)`` ctor builds the per-R image graphs directly.
    //
    // For an owned centre i with reduced neighbours (j, R_j) and (k, R_k) (within the
    // ``r_max_left[itype] + r_max_right[jtype]`` cut, same as get_graph3b), the 3-centre block
    // (gid_j -> gid_k) lives at shift R_k - R_j and is owned by gid_j's row owner; the routing
    // is the same one-Alltoallv idiom as get_graph3b, carrying 5 ints (gj, gk, dRx, dRy, dRz)
    // per block. The self-onsite (j==k, R=0) is dropped — the ImageContainer ctor re-adds the
    // R=0 diagonal. The resulting (gi, gj, R) edge set equals the graph3b pair set the V_nl /
    // force / DM consumers enumerate, so an ImageContainer on this AtomicData has exactly their
    // blocks (and ``build_dmr_images`` can read its pairs straight from the graph).
    AtomicData* get_atomicdata3b(const std::vector<double>& r_max_left,
                                 const std::vector<double>& r_max_right) {
        int initialized = 0;
        MPI_Initialized(&initialized);

        // 1. Reduced neighbour list WITH integer shift (do NOT collapse periodic images).
        std::vector<std::vector<std::pair<int, std::array<int, 3>>>> rnbr(n_atom);
        for (int i = 0; i < n_atom; ++i) {
            const int itype = atom_type[i];
            for (int edge_idx : iconn[i]) {
                const int j = edges[edge_idx].dst;
                double rx, ry, rz;
                get_edge_vec(edge_idx, &rx, &ry, &rz);
                const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                const int jtype = atom_type[j];
                if (r <= r_max_left[itype] + r_max_right[jtype] + 1e-9) {
                    int sx, sy, sz;
                    get_edge_shift_vec(edge_idx, &sx, &sy, &sz);
                    rnbr[i].push_back({get_gid(j), {sx, sy, sz}});
                }
            }
        }

        // 2. Each ordered neighbour pair (j,Rj),(k,Rk) of i -> block (gid_j -> gid_k) at Rk-Rj,
        //    routed to gid_j's owner (5 ints each).
        std::map<int, std::vector<std::array<int, 5>>> send_map;
        for (int i = 0; i < n_atom; ++i) {
            const auto& nb = rnbr[i];
            for (const auto& a : nb) {
                const int owner_j = graph->find_owner(a.first);
                for (const auto& b : nb) {
                    send_map[owner_j].push_back({a.first, b.first,
                        b.second[0] - a.second[0], b.second[1] - a.second[1], b.second[2] - a.second[2]});
                }
            }
        }
        for (auto& kv : send_map) {
            std::sort(kv.second.begin(), kv.second.end());
            kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
        }

        std::vector<int> send_counts(size, 0);
        for (auto& kv : send_map) send_counts[kv.first] = static_cast<int>(kv.second.size()) * 5;
        std::vector<int> recv_counts(size);
        if (initialized) {
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        } else {
            recv_counts = send_counts;
        }
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            sdispls[i + 1] = sdispls[i] + send_counts[i];
            rdispls[i + 1] = rdispls[i] + recv_counts[i];
        }
        std::vector<int> send_buf(sdispls[size]);
        for (auto& kv : send_map) {
            int off = sdispls[kv.first];
            for (const auto& e : kv.second) for (int t = 0; t < 5; ++t) send_buf[off++] = e[t];
        }
        std::vector<int> recv_buf(rdispls[size]);
        if (initialized) {
            MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                          recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, comm);
        } else {
            recv_buf = send_buf;
        }

        // 3. Dedup received 3-centre edges (gid_j owned here); drop the self-onsite.
        std::set<std::array<int, 5>> edge_set;
        for (size_t p = 0; p + 4 < recv_buf.size(); p += 5) {
            if (recv_buf[p] == recv_buf[p + 1] &&
                recv_buf[p + 2] == 0 && recv_buf[p + 3] == 0 && recv_buf[p + 4] == 0) continue;
            edge_set.insert({recv_buf[p], recv_buf[p + 1], recv_buf[p + 2], recv_buf[p + 3], recv_buf[p + 4]});
        }

        // graph3b ⊇ the 2-body S/T graph: add this rank's owned 2-body edges (i -> j, R_j) WITH
        // shift (owned locally, no routing). The 3-centre enumeration alone covers only pairs
        // sharing a common centre, so the pure 2-body pairs must be added explicitly — mirrors
        // get_graph3b's "add two-body and onsite edges". The R=0 onsite is re-added by the
        // ImageContainer ctor, so it is skipped here.
        for (int i = 0; i < n_atom; ++i) {
            const int gi = get_gid(i);
            for (int edge_idx : iconn[i]) {
                const int j = edges[edge_idx].dst;
                int sx, sy, sz;
                get_edge_shift_vec(edge_idx, &sx, &sy, &sz);
                if (gi == get_gid(j) && sx == 0 && sy == 0 && sz == 0) continue;
                edge_set.insert({gi, get_gid(j), sx, sy, sz});
            }
        }

        std::vector<int> edge_index, edge_shift;
        edge_index.reserve(edge_set.size() * 2);
        edge_shift.reserve(edge_set.size() * 3);
        for (const auto& e : edge_set) {
            edge_index.push_back(e[0]); edge_index.push_back(e[1]);
            edge_shift.push_back(e[2]); edge_shift.push_back(e[3]); edge_shift.push_back(e[4]);
        }
        int n_edge3 = static_cast<int>(edge_set.size());
        int N_edge3 = n_edge3;
        if (initialized) MPI_Allreduce(&n_edge3, &N_edge3, 1, MPI_INT, MPI_SUM, comm);

        // 4. Owned per-atom arrays copied from this AtomicData.
        std::vector<int> at_index(n_atom), at_type(n_atom), at_num(n_atom);
        std::vector<double> pos(3 * n_atom);
        for (int i = 0; i < n_atom; ++i) {
            at_index[i] = atom_index[i];
            at_type[i] = atom_type[i];
            at_num[i] = atomic_numbers[i];
            pos[3 * i] = x[i]; pos[3 * i + 1] = y[i]; pos[3 * i + 2] = z[i];
        }

        return new AtomicData(
            static_cast<size_t>(n_atom), static_cast<size_t>(N_atom), static_cast<size_t>(atom_offset),
            static_cast<size_t>(n_edge3), static_cast<size_t>(N_edge3),
            at_index.data(), at_type.data(), edge_index.data(), type_norb.data(), edge_shift.data(),
            cell.data(), pos.data(), at_num.data(), comm, pbc);
    }

private:
    static std::vector<bool> infer_pbc_from_edge_shifts(size_t n_edge, const int *edge_shift_vec_in, int initialized, MPI_Comm comm) {
        int local_flags[3] = {0, 0, 0};
        for (size_t k = 0; k < n_edge; ++k) {
            if (edge_shift_vec_in[3*k] != 0) local_flags[0] = 1;
            if (edge_shift_vec_in[3*k+1] != 0) local_flags[1] = 1;
            if (edge_shift_vec_in[3*k+2] != 0) local_flags[2] = 1;
            if (local_flags[0] && local_flags[1] && local_flags[2]) break;
        }

        int global_flags[3] = {local_flags[0], local_flags[1], local_flags[2]};
        if (initialized) {
            MPI_Allreduce(local_flags, global_flags, 3, MPI_INT, MPI_MAX, comm);
        }

        return {
            global_flags[0] != 0,
            global_flags[1] != 0,
            global_flags[2] != 0
        };
    }

    template<typename T>
    void exchange_attribute(std::vector<T>& data) {
        if (graph == nullptr || size <= 1 ||
            (graph->recv_indices.empty() && graph->send_indices.empty())) {
            return;
        }

        int initialized = 0;
        MPI_Initialized(&initialized);
        
        std::vector<T> send_buf(graph->send_indices.size());
        for(size_t i=0; i<graph->send_indices.size(); ++i) {
            send_buf[i] = data[graph->send_indices[i]];
        }
        
        std::vector<T> recv_buf(graph->recv_indices.size());
        
        size_t type_size = sizeof(T);
        std::vector<int> sdispls_bytes(size + 1);
        std::vector<int> rdispls_bytes(size + 1);
        std::vector<int> send_counts_bytes(size);
        std::vector<int> recv_counts_bytes(size);
        
        for(int i=0; i<size; ++i) {
            send_counts_bytes[i] = graph->send_counts[i] * type_size;
            recv_counts_bytes[i] = graph->recv_counts[i] * type_size;
            sdispls_bytes[i] = graph->send_displs[i] * type_size;
            rdispls_bytes[i] = graph->recv_displs[i] * type_size;
        }
        
        if (initialized) {
            MPI_Alltoallv(send_buf.data(), send_counts_bytes.data(), sdispls_bytes.data(), MPI_BYTE,
                          recv_buf.data(), recv_counts_bytes.data(), rdispls_bytes.data(), MPI_BYTE, comm);
        } else {
            std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
        }
                      
        for(size_t i=0; i<graph->recv_indices.size(); ++i) {
            data[graph->recv_indices[i]] = recv_buf[i];
        }
    }
    
    int get_gid(int lid) const {
        if (lid < n_atom) return graph->owned_global_indices[lid];
        return graph->ghost_global_indices[lid - n_atom];
    }

public:
    // partition_graph is public so its weighting can be unit-tested. It is a
    // pure function of its arguments -- no AtomicData state is touched -- so
    // exposing it costs no encapsulation, and the alternative was a weighting
    // whose only coverage was an end-to-end balance check that passed with
    // the weights disconnected.
    /// Refines a partition for edge cut, balancing VERTEX WEIGHT rather than
    /// vertex count.
    ///
    /// `vwgt` is one weight per owned atom, normally its orbital count: every
    /// cost that matters downstream -- the block rows of H and S, the matvec,
    /// the grid collocation -- scales with orbitals, not with atoms. Balancing
    /// atom count instead leaves a cell of mixed species imbalanced in
    /// proportion to the spread in orbitals per species, silently. Inertial
    /// bisection already weights by the same quantity, so passing it here is
    /// also what stops the refinement from optimising a different objective
    /// than the partition it refines.
    ///
    /// `vwgt` is required, and weighting is unconditional, because wgtflag is
    /// a COLLECTIVE argument: every rank must pass the same value. Deciding it
    /// from whether this rank's weights are empty would flip it on exactly the
    /// ranks that own no atoms -- a cell with fewer atoms than ranks -- and
    /// ParMETIS then hangs on mismatched flags. Same hazard the dummy pointers
    /// below exist for, one level up.
    static void partition_graph(const std::vector<int>& vtxdist, const std::vector<int>& xadj, const std::vector<int>& adjncy, 
                                int nparts, std::vector<int>& part, MPI_Comm comm, const std::vector<double>& pos, int n_global,
                                const std::vector<int>& vwgt) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (nparts <= 1 || !initialized || comm == MPI_COMM_NULL) {
            std::fill(part.begin(), part.end(), 0);
            return;
        }
#ifdef VBCSR_HAVE_PARMETIS
        // ParMETIS Implementation
        // wgtflag 2 = weights on the vertices only (adjwgt stays null).
        // Unconditional: see the note on the signature.
        idx_t wgtflag = 2;
        idx_t numflag = 0; 
        idx_t ncon = 1;    
        idx_t nparts_t = nparts;
        
        std::vector<real_t> tpwgts(ncon * nparts, 1.0 / nparts);
        real_t ubvec[1] = {1.05}; 
        
        idx_t options[3] = {0, 0, 0};
        idx_t edgecut;
        
        MPI_Comm comm_ = comm;
        
        if (vtxdist.back() == 0) return;

        std::vector<idx_t> vtxdist_t(vtxdist.begin(), vtxdist.end());
        std::vector<idx_t> xadj_t(xadj.begin(), xadj.end());
        std::vector<idx_t> adjncy_t(adjncy.begin(), adjncy.end());
        std::vector<idx_t> part_t(part.begin(), part.end());
        
        // ParMETIS rejects null array pointers even where the array is empty,
        // which happens on any rank that owns no vertices -- a cell with fewer
        // atoms than ranks. Point those at a dummy element instead; the rank
        // contributes nothing either way, but the call stays collective.
        idx_t* adjncy_ptr = adjncy_t.empty() ? NULL : adjncy_t.data();
        idx_t dummy_adj = 0;
        if (adjncy_t.empty()) adjncy_ptr = &dummy_adj;

        idx_t* part_ptr = part_t.data();
        idx_t dummy_part = 0;
        if (part_t.empty()) part_ptr = &dummy_part;

        // Weights are clamped to 1: ParMETIS balances a sum of these, and a
        // zero-weight vertex is free to pile up anywhere without registering
        // as imbalance.
        if (vwgt.size() != part.size()) {
            throw std::runtime_error(
                "partition_graph: one vertex weight per owned atom is required.");
        }
        std::vector<idx_t> vwgt_t;
        vwgt_t.reserve(vwgt.size());
        for (int w : vwgt) vwgt_t.push_back(static_cast<idx_t>(std::max(1, w)));
        idx_t dummy_vwgt = 1;
        idx_t* vwgt_ptr = vwgt_t.empty() ? &dummy_vwgt : vwgt_t.data();

        ParMETIS_V3_RefineKway(vtxdist_t.data(), xadj_t.data(), adjncy_ptr,
                               vwgt_ptr, NULL, &wgtflag, &numflag, &ncon, &nparts_t,
                               tpwgts.data(), ubvec, options, &edgecut, part_ptr, &comm_);

                       
        for(size_t i=0; i<part.size(); ++i) part[i] = part_t[i];
#else
        // Hilbert Curve Fallback Implementation
        //
        // Balances atom COUNT, not `vwgt`: it splits a Morton ordering into
        // equal-sized runs. Only reached when ParMETIS is absent, and the
        // difference shows only for a cell of mixed species.
        (void)vwgt;
        int rank, size;
        if (initialized) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }
        
        if (n_global == 0) return;
        
        int my_n_atom = pos.size() / 3;
        
        // 1. Compute Bounding Box
        double min_val[3] = {1e300, 1e300, 1e300};
        double max_val[3] = {-1e300, -1e300, -1e300};
        
        for(int i=0; i<my_n_atom; ++i) {
            for(int d=0; d<3; ++d) {
                if (pos[3*i+d] < min_val[d]) min_val[d] = pos[3*i+d];
                if (pos[3*i+d] > max_val[d]) max_val[d] = pos[3*i+d];
            }
        }
        
        double global_min[3], global_max[3];
        if (initialized) {
            MPI_Allreduce(min_val, global_min, 3, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(max_val, global_max, 3, MPI_DOUBLE, MPI_MAX, comm);
        } else {
            std::copy(min_val, min_val+3, global_min);
            std::copy(max_val, max_val+3, global_max);
        }
        
        // 2. Compute Local Morton Codes
        struct AtomInfo {
            uint64_t code;
            int rank;
            int local_idx;
            
            // For sorting
            bool operator<(const AtomInfo& other) const {
                return code < other.code;
            }
        };
        
        std::vector<AtomInfo> local_infos(my_n_atom);
        double range[3];
        for(int d=0; d<3; ++d) range[d] = global_max[d] - global_min[d] + 1e-9;
        
        uint64_t max_int = (1ULL << 21) - 1;
        
        for(int i=0; i<my_n_atom; ++i) {
            uint64_t x = (uint64_t)((pos[3*i] - global_min[0]) / range[0] * max_int);
            uint64_t y = (uint64_t)((pos[3*i+1] - global_min[1]) / range[1] * max_int);
            uint64_t z = (uint64_t)((pos[3*i+2] - global_min[2]) / range[2] * max_int);
            
            uint64_t code = 0;
            for(int b=0; b<21; ++b) {
                code |= ((x >> b) & 1) << (3*b);
                code |= ((y >> b) & 1) << (3*b + 1);
                code |= ((z >> b) & 1) << (3*b + 2);
            }
            
            local_infos[i] = {code, rank, i};
        }
        
        // 3. Gather all info to Rank 0. AtomInfo is a 16-byte POD
        // (uint64 + 2 ints), transferred as raw MPI_BYTE.
        std::vector<int> recv_counts(size);
        int local_bytes = my_n_atom * sizeof(AtomInfo);
        if (initialized) {
            MPI_Gather(&local_bytes, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, comm);
        } else {
            recv_counts[0] = local_bytes;
        }
        
        std::vector<int> displs(size + 1, 0);
        std::vector<char> recv_buf;
        
        if (rank == 0) {
            for(int i=0; i<size; ++i) displs[i+1] = displs[i] + recv_counts[i];
            recv_buf.resize(displs[size]);
        }
        
        if (initialized) {
            MPI_Gatherv((char*)local_infos.data(), local_bytes, MPI_BYTE,
                        recv_buf.data(), recv_counts.data(), displs.data(), MPI_BYTE, 0, comm);
        } else {
            std::copy((char*)local_infos.data(), (char*)local_infos.data() + local_bytes, recv_buf.data());
        }
                    
        // 4. Sort and Assign (Rank 0)
        // Each AtomInfo carries its (original rank, original local index), so the
        // gathered buffer can be sorted freely and the assignments written back
        // into per-rank vectors ordered by local index, then scattered.
        std::vector<std::vector<int>> per_rank_parts(size);
        if (rank == 0) {
            // First, initialize vectors with correct size
            for(int i=0; i<size; ++i) {
                per_rank_parts[i].resize(recv_counts[i] / sizeof(AtomInfo));
            }

            AtomInfo* all_data = (AtomInfo*)recv_buf.data();
            int total_atoms = displs[size] / sizeof(AtomInfo);

            std::vector<AtomInfo> global_infos(all_data, all_data + total_atoms);
            std::sort(global_infos.begin(), global_infos.end());
            
            // Assign partitions
            int atoms_per_rank = total_atoms / size;
            int remainder = total_atoms % size;
            
            int current_assigned_rank = 0;
            int current_count = 0;
            int target_count = atoms_per_rank + (0 < remainder ? 1 : 0);
            
            for(int i=0; i<total_atoms; ++i) {
                if (current_count >= target_count) {
                    current_assigned_rank++;
                    current_count = 0;
                    target_count = atoms_per_rank + (current_assigned_rank < remainder ? 1 : 0);
                }
                
                int orig_r = global_infos[i].rank;
                int orig_idx = global_infos[i].local_idx;
                
                per_rank_parts[orig_r][orig_idx] = current_assigned_rank;
                current_count++;
            }
        }
        
        // 5. Scatter assignments back
        // Need to flatten per_rank_parts to send buffer, but use Scatterv
        std::vector<int> send_parts_flat;
        std::vector<int> parts_counts(size), parts_displs(size + 1, 0);
        
        if (rank == 0) {
            for(int i=0; i<size; ++i) {
                parts_counts[i] = per_rank_parts[i].size();
                parts_displs[i+1] = parts_displs[i] + parts_counts[i];
                send_parts_flat.insert(send_parts_flat.end(), per_rank_parts[i].begin(), per_rank_parts[i].end());
            }
        }
        
        // Recv buffer is `part`
        if (initialized) {
            MPI_Scatterv(send_parts_flat.data(), parts_counts.data(), parts_displs.data(), MPI_INT,
                         part.data(), my_n_atom, MPI_INT, 0, comm);
        } else {
            std::copy(send_parts_flat.begin(), send_parts_flat.end(), part.begin());
        }
        
#endif
    }

private:

    // Spreads the atoms over the ranks and builds their edges, with no rank
    // ever holding the whole geometry.
    //
    // Rank 0 supplies pos/z; from there the work is distributed throughout:
    //
    //   block scatter -> inertial bisection -> halo exchange -> local search
    //
    // Bisection is what makes the local search possible: once atoms are grouped
    // by region, a rank needs only its own atoms plus the shell of foreign ones
    // within the cutoff, and the search over owned+halo is non-periodic because
    // the halo arrives as explicit shifted images (distributed_build.hpp).
    //
    // Outputs match what the caller's ParMETIS stage expects: `my_indices` is
    // the atom's position in the input, and edge endpoints are in the
    // intermediate numbering where rank r owns [exscan_r, exscan_r + n_r).
    static void distribute_and_build_edges(
        MPI_Comm comm, int rank, int size,
        const std::vector<double>& pos, const std::vector<int>& z,
        const std::vector<double>& cell, const std::vector<bool>& pbc,
        const std::vector<double>& r_max_per_type,
        std::vector<int>& type_norb,
        int& n_global,
        int& my_n_atom,
        std::vector<double>& my_pos,
        std::vector<int>& my_z,
        std::vector<int>& my_types,
        std::vector<int>& my_indices,
        std::vector<int>& my_edges_flat
    ) {
        int initialized = 0;
        MPI_Initialized(&initialized);

        // Types are the distinct atomic numbers in ascending order; every rank
        // needs that list to read the per-type tables the same way.
        std::vector<int> unique_z;
        if (rank == 0) {
            n_global = static_cast<int>(z.size());
            unique_z = z;
            std::sort(unique_z.begin(), unique_z.end());
            unique_z.erase(std::unique(unique_z.begin(), unique_z.end()), unique_z.end());
        }
        if (initialized) MPI_Bcast(&n_global, 1, MPI_INT, 0, comm);
        if (n_global == 0) return;

        int n_types = static_cast<int>(unique_z.size());
        if (initialized) MPI_Bcast(&n_types, 1, MPI_INT, 0, comm);
        unique_z.resize(n_types);
        if (initialized) MPI_Bcast(unique_z.data(), n_types, MPI_INT, 0, comm);
        std::map<int, int> z_to_type;
        for (int i = 0; i < n_types; ++i) z_to_type[unique_z[i]] = i;

        // Callers may pass fewer entries than types (e.g. one uniform cutoff);
        // pad with the last so per-type lookups stay in bounds.
        if (type_norb.empty()) type_norb.assign(n_types, 1);
        else if ((int)type_norb.size() < n_types) type_norb.resize(n_types, type_norb.back());
        std::vector<double> r_max_type = r_max_per_type;
        if (r_max_type.empty()) r_max_type.assign(n_types, 0.0);
        else if ((int)r_max_type.size() < n_types) r_max_type.resize(n_types, r_max_type.back());

        // Block scatter: contiguous chunks of the input, which bisection is
        // about to undo anyway, so there is no point sorting first.
        std::vector<int> counts(size, 0), displs(size + 1, 0);
        for (int r = 0; r < size; ++r) {
            counts[r] = n_global / size + (r < n_global % size ? 1 : 0);
            displs[r + 1] = displs[r] + counts[r];
        }
        const int n_block = counts[rank];

        LocalAtoms mine;
        mine.pos.resize(static_cast<size_t>(n_block) * 3);
        mine.z.resize(n_block);
        if (initialized) {
            std::vector<int> counts3(size), displs3(size + 1, 0);
            for (int r = 0; r < size; ++r) {
                counts3[r] = counts[r] * 3;
                displs3[r + 1] = displs3[r] + counts3[r];
            }
            MPI_Scatterv(pos.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                         mine.pos.data(), n_block * 3, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(z.data(), counts.data(), displs.data(), MPI_INT,
                         mine.z.data(), n_block, MPI_INT, 0, comm);
        } else {
            std::copy(pos.begin(), pos.begin() + n_block * 3, mine.pos.begin());
            std::copy(z.begin(), z.begin() + n_block, mine.z.begin());
        }
        mine.type.resize(n_block);
        mine.global_id.resize(n_block);
        // Balance on orbital count, not atom count: a rank's share of the matrix
        // is blocks, and a 9-orbital atom is nine rows where a 1-orbital atom is
        // one.
        std::vector<double> weights(n_block);
        for (int i = 0; i < n_block; ++i) {
            auto it = z_to_type.find(mine.z[i]);
            if (it == z_to_type.end()) {
                throw std::runtime_error("from_points: atom of species Z=" +
                                         std::to_string(mine.z[i]) + " has no type.");
            }
            mine.type[i] = it->second;
            mine.global_id[i] = displs[rank] + i;
            weights[i] = static_cast<double>(type_norb[it->second]);
        }

        std::unique_ptr<InertialCut> tree;
        LocalAtoms owned = RedistributeByInertia(mine, weights, comm, &tree);
        LocalEdges edges = BuildLocalEdges(owned, *tree, cell, pbc, r_max_type, comm);

        my_n_atom = owned.n_local();
        my_pos = owned.pos;
        my_z = owned.z;
        my_types = owned.type;
        my_indices = owned.global_id;

        // Edges come back in input numbering; the ParMETIS stage wants the
        // intermediate one, where rank r owns a contiguous block. One allgather
        // of (input id, intermediate id) pairs resolves both endpoints, halo
        // atoms included.
        int my_start = 0;
        if (initialized) {
            MPI_Exscan(&my_n_atom, &my_start, 1, MPI_INT, MPI_SUM, comm);
            if (rank == 0) my_start = 0;
        }
        std::vector<int> input_to_inter(static_cast<size_t>(n_global), -1);
        {
            std::vector<int> mine_pairs(static_cast<size_t>(my_n_atom) * 2);
            for (int i = 0; i < my_n_atom; ++i) {
                mine_pairs[2 * static_cast<size_t>(i)] = my_indices[i];
                mine_pairs[2 * static_cast<size_t>(i) + 1] = my_start + i;
            }
            std::vector<int> pair_counts(size, 2 * my_n_atom), pair_displs(size, 0);
            if (initialized) {
                const int mine_n = 2 * my_n_atom;
                MPI_Allgather(&mine_n, 1, MPI_INT, pair_counts.data(), 1, MPI_INT, comm);
            }
            for (int r = 1; r < size; ++r) pair_displs[r] = pair_displs[r - 1] + pair_counts[r - 1];
            std::vector<int> all(static_cast<size_t>(pair_displs.back() + pair_counts.back()));
            if (initialized) {
                MPI_Allgatherv(mine_pairs.data(), 2 * my_n_atom, MPI_INT, all.data(),
                               pair_counts.data(), pair_displs.data(), MPI_INT, comm);
            } else {
                all = mine_pairs;
            }
            for (size_t k = 0; k + 1 < all.size(); k += 2) input_to_inter[all[k]] = all[k + 1];
        }

        my_edges_flat.resize(static_cast<size_t>(edges.n_edge()) * 5);
        for (int e = 0; e < edges.n_edge(); ++e) {
            const int src = input_to_inter[edges.index[2 * static_cast<size_t>(e)]];
            const int dst = input_to_inter[edges.index[2 * static_cast<size_t>(e) + 1]];
            if (src < 0 || dst < 0) {
                throw std::runtime_error("from_points: unmapped atom in an edge.");
            }
            my_edges_flat[5 * static_cast<size_t>(e)] = src;
            my_edges_flat[5 * static_cast<size_t>(e) + 1] = dst;
            my_edges_flat[5 * static_cast<size_t>(e) + 2] = edges.shift[3 * static_cast<size_t>(e)];
            my_edges_flat[5 * static_cast<size_t>(e) + 3] = edges.shift[3 * static_cast<size_t>(e) + 1];
            my_edges_flat[5 * static_cast<size_t>(e) + 4] = edges.shift[3 * static_cast<size_t>(e) + 2];
        }
    }

    static void build_parmetis_graph(
        MPI_Comm comm, int rank, int size, int my_n_atom,
        const std::vector<int>& my_edges_flat,
        std::vector<int>& vtxdist,
        std::vector<int>& xadj,
        std::vector<int>& adjncy,
        int& my_start
    ) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        
        std::vector<int> my_vtxdist(size);
        if (initialized) {
            MPI_Allgather(&my_n_atom, 1, MPI_INT, my_vtxdist.data(), 1, MPI_INT, comm);
        } else {
            std::fill(my_vtxdist.begin(), my_vtxdist.end(), my_n_atom);
        }
        vtxdist.assign(size + 1, 0);
        for(int i=0; i<size; ++i) vtxdist[i+1] = vtxdist[i] + my_vtxdist[i];
        
        my_start = vtxdist[rank];
        xadj.assign(my_n_atom + 1, 0);
        std::vector<std::vector<int>> adj_list(my_n_atom);
        
        int n_edges = my_edges_flat.size() / 5;
        for(int k=0; k<n_edges; ++k) {
            int src = my_edges_flat[5*k];
            int dst = my_edges_flat[5*k+1];
            int lid = src - my_start;
            if (lid < 0 || lid >= my_n_atom) throw std::runtime_error("Edge source mismatch");
            adj_list[lid].push_back(dst);
        }
        
        int current_offset = 0;
        for(int i=0; i<my_n_atom; ++i) {
            xadj[i] = current_offset;
            std::sort(adj_list[i].begin(), adj_list[i].end()); 
            for(int dst : adj_list[i]) adjncy.push_back(dst);
            current_offset += adj_list[i].size();
        }
        xadj[my_n_atom] = current_offset;
    }

    static void redistribute_atoms(
        MPI_Comm comm, int rank, int size, int my_n_atom,
        const std::vector<int>& part,
        const std::vector<double>& my_pos,
        const std::vector<int>& my_z,
        const std::vector<int>& my_types,
        const std::vector<int>& my_indices,
        const std::vector<int>& my_inter_indices,
        std::vector<double>& r_pos,
        std::vector<int>& r_z,
        std::vector<int>& r_types,
        std::vector<int>& r_indices,
        std::vector<int>& r_inter_indices,
        int& total_recv
    ) {
        std::vector<std::vector<int>> atoms_to_send(size);
        for(int i=0; i<my_n_atom; ++i) atoms_to_send[part[i]].push_back(i);
        
        std::vector<int> send_cnts(size), recv_cnts(size);
        for(int i=0; i<size; ++i) send_cnts[i] = atoms_to_send[i].size();
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Alltoall(send_cnts.data(), 1, MPI_INT, recv_cnts.data(), 1, MPI_INT, comm);
        } else {
            std::fill(recv_cnts.begin(), recv_cnts.end(), send_cnts[0]);
        }
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_cnts[i];
            rdispls[i+1] = rdispls[i] + recv_cnts[i];
        }
        
        int total_send = sdispls[size];
        std::vector<double> s_pos(total_send * 3);
        std::vector<int> s_z(total_send);
        std::vector<int> s_types(total_send);
        std::vector<int> s_indices(total_send);
        std::vector<int> s_inter_indices(total_send);
        
        int offset = 0;
        for(int r=0; r<size; ++r) {
            for(int lid : atoms_to_send[r]) {
                s_pos[3*offset] = my_pos[3*lid];
                s_pos[3*offset+1] = my_pos[3*lid+1];
                s_pos[3*offset+2] = my_pos[3*lid+2];
                s_z[offset] = my_z[lid];
                s_types[offset] = my_types[lid];
                s_indices[offset] = my_indices[lid];
                s_inter_indices[offset] = my_inter_indices[lid];
                offset++;
            }
        }
        
        total_recv = rdispls[size];
        r_pos.resize(total_recv * 3);
        r_z.resize(total_recv);
        r_types.resize(total_recv);
        r_indices.resize(total_recv);
        r_inter_indices.resize(total_recv);
        
        std::vector<int> send_cnts_3(size), recv_cnts_3(size), sdispls_3(size+1), rdispls_3(size+1);
        for(int i=0; i<size; ++i) {
            send_cnts_3[i] = send_cnts[i] * 3;
            recv_cnts_3[i] = recv_cnts[i] * 3;
            sdispls_3[i] = sdispls[i] * 3;
            rdispls_3[i] = rdispls[i] * 3;
        }
        sdispls_3[size] = sdispls[size] * 3;
        rdispls_3[size] = rdispls[size] * 3;
        
        if (initialized) {
            MPI_Alltoallv(s_pos.data(), send_cnts_3.data(), sdispls_3.data(), MPI_DOUBLE,
                          r_pos.data(), recv_cnts_3.data(), rdispls_3.data(), MPI_DOUBLE, comm);
            MPI_Alltoallv(s_z.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                          r_z.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
            MPI_Alltoallv(s_types.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                          r_types.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
            MPI_Alltoallv(s_indices.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                          r_indices.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
            MPI_Alltoallv(s_inter_indices.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                          r_inter_indices.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
        } else {
            std::copy(s_pos.begin(), s_pos.end(), r_pos.begin());
            std::copy(s_z.begin(), s_z.end(), r_z.begin());
            std::copy(s_types.begin(), s_types.end(), r_types.begin());
            std::copy(s_indices.begin(), s_indices.end(), r_indices.begin());
            std::copy(s_inter_indices.begin(), s_inter_indices.end(), r_inter_indices.begin());
        }
    }

    static void redistribute_edges(
        MPI_Comm comm, int rank, int size, int my_start,
        const std::vector<int>& my_edges_flat,
        const std::vector<int>& part,
        std::vector<int>& r_edges
    ) {
        std::vector<std::vector<int>> edges_to_send_final(size);
        std::vector<int> lid_to_rank = part;
        int initialized;
        MPI_Initialized(&initialized);
        
        int n_edges = my_edges_flat.size() / 5;
        for(int k=0; k<n_edges; ++k) {
            int src = my_edges_flat[5*k];
            int dst = my_edges_flat[5*k+1];
            int rx = my_edges_flat[5*k+2];
            int ry = my_edges_flat[5*k+3];
            int rz = my_edges_flat[5*k+4];
            
            int lid = src - my_start;
            int target_rank = lid_to_rank[lid];
            
            edges_to_send_final[target_rank].push_back(src);
            edges_to_send_final[target_rank].push_back(dst);
            edges_to_send_final[target_rank].push_back(rx);
            edges_to_send_final[target_rank].push_back(ry);
            edges_to_send_final[target_rank].push_back(rz);
        }
        
        std::vector<int> e_send_cnts(size), e_recv_cnts(size);
        for(int i=0; i<size; ++i) e_send_cnts[i] = edges_to_send_final[i].size();

        if (initialized) {
            MPI_Alltoall(e_send_cnts.data(), 1, MPI_INT, e_recv_cnts.data(), 1, MPI_INT, comm);
        } else {
            std::copy(e_send_cnts.begin(), e_send_cnts.end(), e_recv_cnts.begin());
        }
        
        std::vector<int> e_sdispls(size + 1, 0), e_rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            e_sdispls[i+1] = e_sdispls[i] + e_send_cnts[i];
            e_rdispls[i+1] = e_rdispls[i] + e_recv_cnts[i];
        }
        
        std::vector<int> s_edges(e_sdispls[size]);
        int offset = 0;
        for(int i=0; i<size; ++i) {
            std::copy(edges_to_send_final[i].begin(), edges_to_send_final[i].end(), s_edges.begin() + offset);
            offset += e_send_cnts[i];
        }
        
        r_edges.resize(e_rdispls[size]);
        if (initialized) {
            MPI_Alltoallv(s_edges.data(), e_send_cnts.data(), e_sdispls.data(), MPI_INT,
                          r_edges.data(), e_recv_cnts.data(), e_rdispls.data(), MPI_INT, comm);
        } else {
            std::copy(s_edges.begin(), s_edges.end(), r_edges.begin());
        }
    }

    static AtomicData* construct_final_object(
        MPI_Comm comm, int rank, int size,
        std::vector<double> cell,
        const std::vector<bool>& pbc,
        int total_recv,
        const std::vector<int>& r_indices,
        const std::vector<int>& r_atomic_numbers,
        const std::vector<int>& r_types,
        const std::vector<double>& r_pos,
        const std::vector<int>& r_edges,
        const std::vector<int>& type_norb
    ) {
        if (cell.empty()) cell.resize(9);
        std::vector<bool> final_pbc = pbc;
        if (final_pbc.empty()) final_pbc = {false, false, false};
        if (final_pbc.size() != 3) {
            throw std::runtime_error("pbc must have exactly 3 elements");
        }
        int initialized;
        MPI_Initialized(&initialized);
        
        if (initialized) {
            MPI_Bcast(cell.data(), 9, MPI_DOUBLE, 0, comm);
        }

        int pbc_values[3] = {
            final_pbc[0] ? 1 : 0,
            final_pbc[1] ? 1 : 0,
            final_pbc[2] ? 1 : 0
        };
        if (initialized) {
            MPI_Bcast(pbc_values, 3, MPI_INT, 0, comm);
        }
        final_pbc = {
            pbc_values[0] != 0,
            pbc_values[1] != 0,
            pbc_values[2] != 0
        };
        
        int my_final_n = total_recv;
        std::vector<int> final_counts(size);
        if (initialized) {
            MPI_Allgather(&my_final_n, 1, MPI_INT, final_counts.data(), 1, MPI_INT, comm);
        } else {
            final_counts[0] = my_final_n;
        }
        
        int my_final_offset = 0;
        for(int i=0; i<rank; ++i) my_final_offset += final_counts[i];
        
        int total_atoms = 0;
        for(int c : final_counts) total_atoms += c;
        
        int my_final_n_edge = r_edges.size() / 5;
        int total_edges_global = 0;
        if (initialized) {
            MPI_Allreduce(&my_final_n_edge, &total_edges_global, 1, MPI_INT, MPI_SUM, comm);
        } else {
            total_edges_global = my_final_n_edge;
        }
        
        std::vector<int> edge_indices(my_final_n_edge * 2);
        std::vector<int> edge_shifts(my_final_n_edge * 3);
        
        for(int k=0; k<my_final_n_edge; ++k) {
            edge_indices[2*k] = r_edges[5*k];
            edge_indices[2*k+1] = r_edges[5*k+1];
            edge_shifts[3*k] = r_edges[5*k+2];
            edge_shifts[3*k+1] = r_edges[5*k+3];
            edge_shifts[3*k+2] = r_edges[5*k+4];
        }
        
        return new AtomicData(my_final_n, total_atoms, my_final_offset, my_final_n_edge, total_edges_global,
                              r_indices.data(), r_types.data(), edge_indices.data(), type_norb.data(), edge_shifts.data(),
                              cell.data(), r_pos.data(), r_atomic_numbers.data(), comm, final_pbc);
    }

};

}
}
