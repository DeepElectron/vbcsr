// Distributed atomic-graph construction against the serial reference.
//
// The contract is equivalence, not approximation: the union of the per-rank
// edge sets must equal, as a set, the edge list a single-rank NeighborList
// produces over the same geometry. Anything else -- a halo one shell too
// narrow, a periodic image counted twice, a bond dropped at a partition
// boundary -- shows up here as a set difference.
//
// Run under MPI to exercise the distributed path.

#include "vbcsr/core/atomic/distributed_build.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace {

using vbcsr::atomic::BuildLocalEdges;
using vbcsr::atomic::LocalAtoms;
using vbcsr::atomic::RedistributeByInertia;

int Rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
int Size() { int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }

/// A simple-cubic block of `n` atoms per side, two alternating types.
struct Lattice {
    std::vector<double> pos;
    std::vector<int> z, type;
    std::vector<double> cell;
    int n_atom = 0;

    explicit Lattice(int n, double a = 2.5) {
        cell = {n * a, 0, 0, 0, n * a, 0, 0, 0, n * a};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    pos.insert(pos.end(), {i * a, j * a, k * a});
                    const int t = (i + j + k) % 2;
                    type.push_back(t);
                    z.push_back(t == 0 ? 6 : 14);
                }
        n_atom = static_cast<int>(z.size());
    }
};

using EdgeKey = std::tuple<int, int, int, int, int>;

/// Every edge, built serially on one rank -- the reference.
std::set<EdgeKey> SerialEdges(const Lattice& lat, const std::vector<double>& r_max,
                              const std::vector<bool>& pbc) {
    double rm = 0.0;
    for (double r : r_max) rm = std::max(rm, r);
    vbcsr::atomic::NeighborList nl;
    nl.build(lat.pos, lat.cell, pbc, 2.0 * rm);

    std::set<EdgeKey> out;
    for (int i = 0; i < lat.n_atom; ++i) {
        for (const auto& nb : nl.neighbors[i]) {
            const int j = nb.index;
            const double rc = r_max[lat.type[i]] + r_max[lat.type[j]];
            const double dx = lat.pos[3 * j] - lat.pos[3 * i] +
                              nb.rx * lat.cell[0] + nb.ry * lat.cell[3] + nb.rz * lat.cell[6];
            const double dy = lat.pos[3 * j + 1] - lat.pos[3 * i + 1] +
                              nb.rx * lat.cell[1] + nb.ry * lat.cell[4] + nb.rz * lat.cell[7];
            const double dz = lat.pos[3 * j + 2] - lat.pos[3 * i + 2] +
                              nb.rx * lat.cell[2] + nb.ry * lat.cell[5] + nb.rz * lat.cell[8];
            if (dx * dx + dy * dy + dz * dz > (rc + 1e-9) * (rc + 1e-9)) continue;
            out.insert({i, j, nb.rx, nb.ry, nb.rz});
        }
    }
    return out;
}

/// This rank's contiguous slice, i.e. the naive input distribution.
LocalAtoms BlockSlice(const Lattice& lat) {
    const int n = lat.n_atom, size = Size(), rank = Rank();
    const int base = n / size, rem = n % size;
    const int first = rank * base + std::min(rank, rem);
    const int count = base + (rank < rem ? 1 : 0);
    LocalAtoms out;
    out.pos.assign(lat.pos.begin() + 3 * static_cast<size_t>(first),
                   lat.pos.begin() + 3 * static_cast<size_t>(first + count));
    out.z.assign(lat.z.begin() + first, lat.z.begin() + first + count);
    out.type.assign(lat.type.begin() + first, lat.type.begin() + first + count);
    out.global_id.resize(count);
    for (int i = 0; i < count; ++i) out.global_id[i] = first + i;
    return out;
}

/// Gathers every rank's edges into one set on every rank.
std::set<EdgeKey> GatherEdges(const vbcsr::atomic::LocalEdges& local) {
    std::vector<int> flat;
    for (int e = 0; e < local.n_edge(); ++e) {
        flat.insert(flat.end(), {local.index[2 * e], local.index[2 * e + 1],
                                 local.shift[3 * e], local.shift[3 * e + 1],
                                 local.shift[3 * e + 2]});
    }
    int my_n = static_cast<int>(flat.size());
    std::vector<int> counts(Size(), 0);
    MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(Size(), 0);
    for (int r = 1; r < Size(); ++r) displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<int> all(displs.back() + counts.back());
    MPI_Allgatherv(flat.data(), my_n, MPI_INT, all.data(), counts.data(), displs.data(),
                   MPI_INT, MPI_COMM_WORLD);

    std::set<EdgeKey> out;
    for (size_t k = 0; k + 4 < all.size(); k += 5) {
        out.insert({all[k], all[k + 1], all[k + 2], all[k + 3], all[k + 4]});
    }
    return out;
}

void CheckEquivalence(const Lattice& lat, const std::vector<double>& r_max,
                      const std::vector<bool>& pbc) {
    LocalAtoms mine = BlockSlice(lat);
    std::unique_ptr<vbcsr::atomic::InertialCut> tree;
    LocalAtoms owned = RedistributeByInertia(mine, {}, MPI_COMM_WORLD, &tree);

    // Migration must conserve atoms.
    int n_owned = owned.n_local(), n_total = 0;
    MPI_Allreduce(&n_owned, &n_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(n_total, lat.n_atom);

    const auto local = BuildLocalEdges(owned, *tree, lat.cell, pbc, r_max, MPI_COMM_WORLD);
    const std::set<EdgeKey> got = GatherEdges(local);
    const std::set<EdgeKey> want = SerialEdges(lat, r_max, pbc);

    std::vector<EdgeKey> missing, extra;
    std::set_difference(want.begin(), want.end(), got.begin(), got.end(),
                        std::back_inserter(missing));
    std::set_difference(got.begin(), got.end(), want.begin(), want.end(),
                        std::back_inserter(extra));
    EXPECT_TRUE(missing.empty()) << missing.size() << " edges missing, first ("
                                 << std::get<0>(missing[0]) << "," << std::get<1>(missing[0])
                                 << ";" << std::get<2>(missing[0]) << ","
                                 << std::get<3>(missing[0]) << "," << std::get<4>(missing[0]) << ")";
    EXPECT_TRUE(extra.empty()) << extra.size() << " spurious edges, first ("
                               << std::get<0>(extra[0]) << "," << std::get<1>(extra[0])
                               << ";" << std::get<2>(extra[0]) << "," << std::get<3>(extra[0])
                               << "," << std::get<4>(extra[0]) << ")";
    EXPECT_EQ(got.size(), want.size());
}

}  // namespace

// Nearest neighbours only: the halo is thin, so a boundary atom needs exactly
// one shell of foreign atoms.
TEST(DistributedBuild, MatchesSerialShortCutoff) {
    Lattice lat(6);
    CheckEquivalence(lat, {1.6, 1.6}, {true, true, true});
}

// Type-dependent reach: pairs bond within r_i + r_j, so the two species see
// different neighbourhoods and a halo sized by the max is required.
TEST(DistributedBuild, MatchesSerialMixedCutoffs) {
    Lattice lat(6);
    CheckEquivalence(lat, {1.4, 3.2}, {true, true, true});
}

// A cutoff comparable to the cell: several periodic images of the same atom
// pair up, which is where an image counted twice would surface.
TEST(DistributedBuild, MatchesSerialLongCutoffManyImages) {
    Lattice lat(4);
    CheckEquivalence(lat, {3.0, 3.0}, {true, true, true});
}

// Open boundaries: no images at all, so any shift other than zero is a bug.
TEST(DistributedBuild, MatchesSerialNonPeriodic) {
    Lattice lat(5);
    CheckEquivalence(lat, {2.6, 2.6}, {false, false, false});
}

// A slab: periodic in-plane, open along z. Thin directions generate more
// image shells than thick ones, which the shift search has to size per axis.
TEST(DistributedBuild, MatchesSerialSlab) {
    Lattice lat(5);
    lat.cell[8] = 40.0;  // vacuum along z
    CheckEquivalence(lat, {2.6, 2.6}, {true, true, false});
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
