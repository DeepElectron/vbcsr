// from_points builds its geometry distributed. This asserts it still produces
// the same graph.
//
// The contract is equivalence, not approximation: the union over ranks of the
// edges from_points ends up holding, written back in input-atom numbering, must
// equal as a set what a single serial NeighborList produces over the same
// geometry. A halo one shell too narrow, a periodic image counted twice, a bond
// dropped at a partition boundary, or a spurious self-edge all show up here as a
// set difference -- and equally, the answer must not depend on the rank count.

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace {

using vbcsr::atomic::AtomicData;

int Rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
int Size() { int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }

/// A simple-cubic block of `n` atoms per side, two alternating species.
struct Lattice {
    std::vector<double> pos;
    std::vector<int> z;
    std::vector<double> cell;
    std::vector<bool> pbc{true, true, true};
    int n_atom = 0;

    explicit Lattice(int n, double a = 2.5) {
        cell = {n * a, 0, 0, 0, n * a, 0, 0, 0, n * a};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    pos.insert(pos.end(), {i * a, j * a, k * a});
                    z.push_back((i + j + k) % 2 == 0 ? 6 : 14);
                }
        n_atom = static_cast<int>(z.size());
    }

    /// Type index, i.e. position in the ascending distinct-Z list.
    int type_of(int atom) const { return z[atom] == 6 ? 0 : 1; }
};

using EdgeKey = std::tuple<int, int, int, int, int>;

/// Every edge, from one serial neighbour list -- the reference.
std::set<EdgeKey> SerialEdges(const Lattice& lat, const std::vector<double>& r_max) {
    double rm = 0.0;
    for (double r : r_max) rm = std::max(rm, r);
    vbcsr::atomic::NeighborList nl;
    nl.build(lat.pos, lat.cell, lat.pbc, 2.0 * rm);

    std::set<EdgeKey> out;
    for (int i = 0; i < lat.n_atom; ++i) {
        for (const auto& nb : nl.neighbors[i]) {
            const int j = nb.index;
            const double rc = r_max[lat.type_of(i)] + r_max[lat.type_of(j)];
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

/// Every rank's edges, in input-atom numbering, gathered onto every rank.
///
/// AtomicData indexes atoms by their place in the partition; atom_index maps
/// that back to the input for owned atoms and ghosts alike, which is what makes
/// the comparison rank-count independent.
std::set<EdgeKey> GraphEdges(const AtomicData& data) {
    std::vector<int> flat;
    for (int e = 0; e < data.n_edge; ++e) {
        const auto& edge = data.edges[e];
        flat.insert(flat.end(), {data.atom_index[edge.src], data.atom_index[edge.dst],
                                 edge.rx, edge.ry, edge.rz});
    }
    int my_n = static_cast<int>(flat.size());
    std::vector<int> counts(Size(), 0), displs(Size(), 0);
    MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < Size(); ++r) displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<int> all(static_cast<size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(flat.data(), my_n, MPI_INT, all.data(), counts.data(), displs.data(),
                   MPI_INT, MPI_COMM_WORLD);

    std::set<EdgeKey> out;
    for (size_t k = 0; k + 4 < all.size(); k += 5) {
        out.insert({all[k], all[k + 1], all[k + 2], all[k + 3], all[k + 4]});
    }
    return out;
}

void CheckEquivalence(const Lattice& lat, const std::vector<double>& r_max) {
    const std::vector<int> type_norb{1, 4};
    std::unique_ptr<AtomicData> data(AtomicData::from_points(
        lat.pos, lat.z, lat.cell, lat.pbc, r_max, type_norb, MPI_COMM_WORLD));

    // Every atom is owned exactly once.
    int n_owned = data->n_atom, n_total = 0;
    MPI_Allreduce(&n_owned, &n_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(n_total, lat.n_atom);

    const std::set<EdgeKey> got = GraphEdges(*data);
    const std::set<EdgeKey> want = SerialEdges(lat, r_max);

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
TEST(FromPoints, MatchesSerialShortCutoff) {
    CheckEquivalence(Lattice(6), {1.6, 1.6});
}

// Type-dependent reach: pairs bond within r_i + r_j, so the two species see
// different neighbourhoods and a halo sized by the max is required.
TEST(FromPoints, MatchesSerialMixedCutoffs) {
    CheckEquivalence(Lattice(6), {1.4, 3.2});
}

// A cutoff comparable to the cell: several periodic images of the same pair
// appear, which is where an image counted twice would surface.
TEST(FromPoints, MatchesSerialLongCutoffManyImages) {
    CheckEquivalence(Lattice(4), {3.0, 3.0});
}

// Fewer atoms than ranks once the run is wide enough: empty partitions must not
// deadlock or lose bonds.
TEST(FromPoints, MatchesSerialTinyCell) {
    CheckEquivalence(Lattice(2), {1.6, 1.6});
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
