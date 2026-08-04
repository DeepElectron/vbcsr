#ifndef VBCSR_CORE_ATOMIC_DISTRIBUTED_BUILD_HPP
#define VBCSR_CORE_ATOMIC_DISTRIBUTED_BUILD_HPP

// Building the atomic graph without ever assembling it on one rank.
//
// AtomicData::from_points does the whole geometric phase on rank 0: it builds
// the neighbour list over every atom, then buckets every edge per destination
// before scattering. Both objects scale as N * <neighbours>, and the neighbour
// count is set by the cutoff, not by the rank count -- for a 3D cell at a
// two-centre cutoff it is a few hundred. That is fine to ~1e5 atoms and hits a
// wall well before 1e7: at 1e6 atoms and 296 neighbours the edge list alone is
// 5.8 GB on rank 0, transiently doubled by the send buckets, and the serial
// build takes ~14 s.
//
// The route here never forms those. Atoms arrive already spread over the ranks
// (any distribution -- a block split of the input file is the usual one), and:
//
//   1. Recursive inertial bisection decides ownership from the positions
//      alone, in O(log P) rounds of kilobyte-sized collectives.
//   2. Atoms migrate to their owners in one MPI_Alltoallv.
//   3. Each rank pulls a halo: the atoms of other ranks that lie within the
//      cutoff of its own, sent as explicit images so periodicity is already
//      resolved in the coordinates.
//   4. Each rank builds a neighbour list over (its atoms + halo) and keeps the
//      edges whose source it owns.
//
// Every stage is O(N/P) in time and memory. The peak per rank is the local
// neighbour list, which is the answer itself rather than a staging copy.
//
// Purely geometric, like the rest of this directory: positions, per-type
// cutoffs and a part count in, a distributed graph out. What the cutoffs mean
// is the caller's business.

#include "vbcsr/core/atomic/inertial_partition.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>

namespace vbcsr {
namespace atomic {

/// One rank's share of a distributed atom set.
///
/// `global_id` is the atom's index in the caller's original ordering, which is
/// what a Hamiltonian file's indexing refers to; it survives migration so the
/// caller can still map its own data onto the result.
struct LocalAtoms {
    std::vector<double> pos;        ///< 3 * n_local, row-major Cartesian.
    std::vector<int> z;             ///< n_local atomic numbers.
    std::vector<int> type;          ///< n_local type indices.
    std::vector<int> global_id;     ///< n_local original indices.
    int n_local() const { return static_cast<int>(z.size()); }
};

/// Edges whose source atom this rank owns, in global indices.
///
/// `shift` is the lattice translation of the *target*, expressed against the
/// positions as given -- the same convention NeighborList reports and the
/// AtomicData constructors expect, so a bond reads r_j + R - r_i either way.
struct LocalEdges {
    std::vector<int> index;  ///< 2 per edge: global i, global j.
    std::vector<int> shift;  ///< 3 per edge.
    int n_edge() const { return static_cast<int>(index.size() / 2); }
};

namespace detail {

inline int CommRank(MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return 0;
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

inline int CommSize(MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return 1;
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}

/// Moves each element to the rank named by `dest`, for a fixed stride.
template <typename T>
inline std::vector<T> Alltoallv(const std::vector<T>& src, const std::vector<int>& dest,
                                int stride, MPI_Datatype type, MPI_Comm comm) {
    const int size = CommSize(comm);
    const int n = static_cast<int>(dest.size());

    std::vector<int> send_counts(size, 0);
    for (int d : dest) send_counts[d] += stride;
    std::vector<int> recv_counts(size, 0);
    if (comm != MPI_COMM_NULL) {
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    } else {
        recv_counts = send_counts;
    }

    std::vector<int> send_displs(size, 0), recv_displs(size, 0);
    for (int r = 1; r < size; ++r) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }
    const int total_recv = recv_displs.back() + recv_counts.back();

    // Pack in destination order.
    std::vector<int> cursor = send_displs;
    std::vector<T> packed(static_cast<size_t>(send_displs.back() + send_counts.back()));
    for (int i = 0; i < n; ++i) {
        T* out = packed.data() + cursor[dest[i]];
        std::copy(src.begin() + static_cast<size_t>(i) * stride,
                  src.begin() + static_cast<size_t>(i + 1) * stride, out);
        cursor[dest[i]] += stride;
    }

    std::vector<T> received(static_cast<size_t>(total_recv));
    if (comm != MPI_COMM_NULL) {
        MPI_Alltoallv(packed.data(), send_counts.data(), send_displs.data(), type,
                      received.data(), recv_counts.data(), recv_displs.data(), type, comm);
    } else {
        received = packed;
    }
    return received;
}

/// Lattice translations that can bring a neighbour within `cutoff`.
///
/// The search is over the images of the *cell*, so it costs nothing per atom.
/// A cell thinner than the cutoff simply yields more shells.
inline std::vector<std::array<int, 3>> ImageShifts(const std::vector<double>& cell,
                                                   const std::vector<bool>& pbc,
                                                   double cutoff) {
    // Perpendicular width of each lattice direction, so a thin slab does not
    // get the same shell count as a cube.
    int n_shell[3] = {0, 0, 0};
    for (int d = 0; d < 3; ++d) {
        if (!pbc[d]) continue;
        const double* a = &cell[3 * ((d + 1) % 3)];
        const double* b = &cell[3 * ((d + 2) % 3)];
        const double cross[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                 a[0] * b[1] - a[1] * b[0]};
        const double area = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                                      cross[2] * cross[2]);
        const double* c = &cell[3 * d];
        const double volume = std::abs(c[0] * cross[0] + c[1] * cross[1] + c[2] * cross[2]);
        const double width = (area > 0.0) ? volume / area : 0.0;
        n_shell[d] = (width > 0.0) ? static_cast<int>(std::ceil(cutoff / width)) : 0;
    }

    std::vector<std::array<int, 3>> shifts;
    for (int i = -n_shell[0]; i <= n_shell[0]; ++i) {
        for (int j = -n_shell[1]; j <= n_shell[1]; ++j) {
            for (int k = -n_shell[2]; k <= n_shell[2]; ++k) {
                shifts.push_back({i, j, k});
            }
        }
    }
    return shifts;
}

}  // namespace detail

/// Assigns atoms to ranks by inertial bisection and migrates them there.
///
/// Args:
///   local: This rank's atoms, in any distribution.
///   weights: Per-atom work, empty for uniform. Balancing orbital counts
///     rather than atoms is the usual reason to pass them.
///   comm: Ranks taking part.
///   tree_out: The cut tree, identical on every rank, for classifying other
///     objects into the same regions later. Optional.
/// Returns:
///   The atoms this rank now owns.
inline LocalAtoms RedistributeByInertia(const LocalAtoms& local,
                                        const std::vector<double>& weights,
                                        MPI_Comm comm,
                                        std::unique_ptr<InertialCut>* tree_out = nullptr) {
    const int size = detail::CommSize(comm);
    InertialPartition part =
        PartitionByInertiaDistributed(local.pos, weights, size, comm);
    if (tree_out != nullptr) *tree_out = std::move(part.tree);

    LocalAtoms moved;
    moved.pos = detail::Alltoallv(local.pos, part.owner, 3, MPI_DOUBLE, comm);
    moved.z = detail::Alltoallv(local.z, part.owner, 1, MPI_INT, comm);
    moved.type = detail::Alltoallv(local.type, part.owner, 1, MPI_INT, comm);
    moved.global_id = detail::Alltoallv(local.global_id, part.owner, 1, MPI_INT, comm);
    return moved;
}

/// Builds the edges whose source atom this rank owns.
///
/// Each rank first pulls a halo: every owned atom is offered to the ranks whose
/// region comes within `cutoff` of it, as an explicit image at its shifted
/// coordinates. Periodicity is therefore already baked into the halo positions,
/// and the local neighbour search runs non-periodically over owned + halo --
/// which is what keeps an atom from being paired with its own image twice.
///
/// Args:
///   owned: This rank's atoms, as returned by RedistributeByInertia.
///   tree: The cut tree those atoms were partitioned by.
///   cell, pbc: Lattice.
///   r_max_per_type: Per-type reach; a pair (i, j) bonds within r_i + r_j.
///   comm: Ranks taking part.
/// Returns:
///   Edges in global indices, source-owned.
inline LocalEdges BuildLocalEdges(const LocalAtoms& owned,
                                  const InertialCut& tree,
                                  const std::vector<double>& cell,
                                  const std::vector<bool>& pbc,
                                  const std::vector<double>& r_max_per_type,
                                  MPI_Comm comm) {
    const int rank = detail::CommRank(comm);
    const int size = detail::CommSize(comm);
    double r_max = 0.0;
    for (double r : r_max_per_type) r_max = std::max(r_max, r);
    const double cutoff = 2.0 * r_max;

    // --- offer every owned atom to the ranks its cutoff sphere can reach ---
    //
    // An atom is needed by rank q if some point within `cutoff` of it belongs
    // to q, which MarkRanksWithinRadius answers exactly by descending the same
    // cut planes the atoms were partitioned by.
    const auto shifts = detail::ImageShifts(cell, pbc, cutoff);
    std::vector<double> send_pos;
    std::vector<int> send_meta;  // global_id, type, shift(3)
    std::vector<int> dest;

    std::vector<char> wanted(size, 0);
    for (int i = 0; i < owned.n_local(); ++i) {
        for (const auto& s : shifts) {
            const double p[3] = {
                owned.pos[3 * static_cast<size_t>(i)] + s[0] * cell[0] + s[1] * cell[3] + s[2] * cell[6],
                owned.pos[3 * static_cast<size_t>(i) + 1] + s[0] * cell[1] + s[1] * cell[4] + s[2] * cell[7],
                owned.pos[3 * static_cast<size_t>(i) + 2] + s[0] * cell[2] + s[1] * cell[5] + s[2] * cell[8]};

            std::fill(wanted.begin(), wanted.end(), 0);
            MarkRanksWithinRadius(tree, p, cutoff, wanted);

            for (int q = 0; q < size; ++q) {
                // The rank's own atoms are already present; only images of
                // itself (a shift that is not the identity) need sending back.
                const bool identity = (s[0] == 0 && s[1] == 0 && s[2] == 0);
                if (!wanted[q] || (q == rank && identity)) continue;
                send_pos.insert(send_pos.end(), {p[0], p[1], p[2]});
                send_meta.insert(send_meta.end(),
                                 {owned.global_id[i], owned.type[i], s[0], s[1], s[2]});
                dest.push_back(q);
            }
        }
    }

    const std::vector<double> halo_pos = detail::Alltoallv(send_pos, dest, 3, MPI_DOUBLE, comm);
    const std::vector<int> halo_meta = detail::Alltoallv(send_meta, dest, 5, MPI_INT, comm);
    const int n_halo = static_cast<int>(halo_meta.size() / 5);

    // --- neighbour search over owned + halo, without periodicity ---
    //
    // The halo already carries each image at its shifted coordinates, so a
    // periodic search here would find the same pair a second time.
    const int n_own = owned.n_local();
    std::vector<double> all_pos = owned.pos;
    all_pos.insert(all_pos.end(), halo_pos.begin(), halo_pos.end());

    std::vector<int> all_gid(owned.global_id);
    std::vector<int> all_type(owned.type);
    std::vector<std::array<int, 3>> all_shift(n_own, {0, 0, 0});
    for (int h = 0; h < n_halo; ++h) {
        all_gid.push_back(halo_meta[5 * static_cast<size_t>(h)]);
        all_type.push_back(halo_meta[5 * static_cast<size_t>(h) + 1]);
        all_shift.push_back({halo_meta[5 * static_cast<size_t>(h) + 2],
                             halo_meta[5 * static_cast<size_t>(h) + 3],
                             halo_meta[5 * static_cast<size_t>(h) + 4]});
    }

    NeighborList nl;
    nl.build(all_pos, cell, {false, false, false}, cutoff);

    LocalEdges edges;
    for (int i = 0; i < n_own; ++i) {
        // No (i, i, R=0) edge: NeighborList omits the home-image self-pair and
        // so does the serial route, because the R=0 diagonal is re-added when
        // the images are built. Emitting it here would double every onsite block.
        for (const auto& nb : nl.neighbors[i]) {
            const int j = nb.index;
            const double rc = r_max_per_type[all_type[i]] + r_max_per_type[all_type[j]];
            const double dx = all_pos[3 * static_cast<size_t>(j)] - all_pos[3 * static_cast<size_t>(i)];
            const double dy = all_pos[3 * static_cast<size_t>(j) + 1] - all_pos[3 * static_cast<size_t>(i) + 1];
            const double dz = all_pos[3 * static_cast<size_t>(j) + 2] - all_pos[3 * static_cast<size_t>(i) + 2];
            if (dx * dx + dy * dy + dz * dz > (rc + 1e-9) * (rc + 1e-9)) continue;
            edges.index.insert(edges.index.end(), {all_gid[i], all_gid[j]});
            edges.shift.insert(edges.shift.end(),
                               {all_shift[j][0], all_shift[j][1], all_shift[j][2]});
        }
    }
    return edges;
}

}  // namespace atomic
}  // namespace vbcsr

#endif  // VBCSR_CORE_ATOMIC_DISTRIBUTED_BUILD_HPP
