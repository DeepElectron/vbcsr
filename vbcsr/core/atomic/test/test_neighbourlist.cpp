// Test assertions must stay active in Release builds.
#undef NDEBUG

#include "../neighbourlist.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <array>
#include <random>

using namespace vbcsr;
using namespace vbcsr::atomic;

void test_simple_cubic_pbc() {
    std::cout << "Testing Simple Cubic PBC..." << std::endl;
    // 2x2x2 grid of atoms in a 2x2x2 box. Spacing 1.0.
    // Box size 2.0.
    // Cutoff 1.1 (should find nearest neighbors)
    
    std::vector<double> positions;
    for(int z=0; z<2; ++z) {
        for(int y=0; y<2; ++y) {
            for(int x=0; x<2; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {2.0,0,0, 0,2.0,0, 0,0,2.0};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Each atom should have 6 neighbors (up, down, left, right, front, back)
    // due to PBC.
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 6) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 6." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_non_pbc() {
    std::cout << "Testing Non-PBC..." << std::endl;
    // 3 atoms in a line: 0.0, 1.0, 2.0
    // Cutoff 1.1
    // 0-1 connected, 1-2 connected. 0-2 not connected.
    
    std::vector<double> positions = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0
    };
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {false, false, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Atom 0: neighbor 1
    // Atom 1: neighbors 0, 2
    // Atom 2: neighbor 1
    
    assert(nl.neighbors[0].size() == 1);
    assert(nl.neighbors[0][0].index == 1);
    assert(nl.neighbors[0][0].rx == 0 && nl.neighbors[0][0].ry == 0 && nl.neighbors[0][0].rz == 0);
    
    assert(nl.neighbors[1].size() == 2);
    // Order not guaranteed, check existence
    bool has0 = false, has2 = false;
    for(const auto& n : nl.neighbors[1]) {
        if (n.index == 0) has0 = true;
        if (n.index == 2) has2 = true;
    }
    assert(has0 && has2);
    
    assert(nl.neighbors[2].size() == 1);
    assert(nl.neighbors[2][0].index == 1);
    
    std::cout << "Passed." << std::endl;
}

void test_large_system() {
    std::cout << "Testing Large System (1000 atoms)..." << std::endl;
    // 10x10x10 grid
    int N = 10;
    std::vector<double> positions;
    for(int z=0; z<N; ++z) {
        for(int y=0; y<N; ++y) {
            for(int x=0; x<N; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 6) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 6." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_mixed_pbc() {
    std::cout << "Testing Mixed PBC (XY)..." << std::endl;
    // 2x2x2 grid. PBC in X and Y, not Z.
    // Atoms at z=0 and z=1.
    // Box size 2.0.
    // Cutoff 1.1.
    
    std::vector<double> positions;
    for(int z=0; z<2; ++z) {
        for(int y=0; y<2; ++y) {
            for(int x=0; x<2; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {2.0,0,0, 0,2.0,0, 0,0,2.0};
    std::vector<bool> pbc = {true, true, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // For z=0 atoms:
    // Neighbors in X: left, right (wrapped) -> 2
    // Neighbors in Y: front, back (wrapped) -> 2
    // Neighbors in Z: up (z=1) -> 1. Down is boundary (no wrap).
    // Total: 5.
    
    // For z=1 atoms:
    // Neighbors in X: 2
    // Neighbors in Y: 2
    // Neighbors in Z: down (z=0) -> 1. Up is boundary (no wrap).
    // Total: 5.
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 5) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 5." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_exact_indices() {
    std::cout << "Testing Exact Indices..." << std::endl;
    // 3 atoms: 0 at (0,0,0), 1 at (1,0,0), 2 at (0,1,0)
    // Box 10. PBC false. Cutoff 1.1.
    // 0-1 (dist 1), 0-2 (dist 1), 1-2 (dist sqrt(2) ~ 1.41 > 1.1)
    
    std::vector<double> positions = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {false, false, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Atom 0: neighbors 1, 2
    assert(nl.neighbors[0].size() == 2);
    std::vector<int> n0;
    for(const auto& n : nl.neighbors[0]) n0.push_back(n.index);
    std::sort(n0.begin(), n0.end());
    assert(n0[0] == 1 && n0[1] == 2);
    
    // Atom 1: neighbor 0
    assert(nl.neighbors[1].size() == 1);
    assert(nl.neighbors[1][0].index == 0);
    
    // Atom 2: neighbor 0
    assert(nl.neighbors[2].size() == 1);
    assert(nl.neighbors[2][0].index == 0);
    
    std::cout << "Passed." << std::endl;
}

void test_robustness() {
    std::cout << "Testing Robustness..." << std::endl;
    
    std::vector<double> positions = {0,0,0};
    std::vector<double> cell = {1,0,0, 0,1,0, 0,0,1};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.0;
    
    NeighborList nl;
    
    // 1. Invalid positions size
    try {
        std::vector<double> bad_pos = {0,0};
        nl.build(bad_pos, cell, pbc, cutoff);
        std::cerr << "Failed to catch invalid positions size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 2. Invalid cell size
    try {
        std::vector<double> bad_cell = {1,0,0};
        nl.build(positions, bad_cell, pbc, cutoff);
        std::cerr << "Failed to catch invalid cell size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 3. Invalid PBC size
    try {
        std::vector<bool> bad_pbc = {true, true};
        nl.build(positions, cell, bad_pbc, cutoff);
        std::cerr << "Failed to catch invalid PBC size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 4. Zero Volume
    try {
        std::vector<double> zero_cell = {0,0,0, 0,0,0, 0,0,0};
        nl.build(positions, zero_cell, pbc, cutoff);
        std::cerr << "Failed to catch zero volume." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 5. Negative Volume (Determinant is negative, but we take abs, so it might pass if valid shape)
    // Actually, we check abs(vol) < 1e-8.
    // If we give a valid cell with negative determinant, it should work (just handedness change).
    // Let's test a flattened cell (vol=0).
    try {
        std::vector<double> flat_cell = {1,0,0, 1,0,0, 0,0,1}; // Collinear a and b
        nl.build(positions, flat_cell, pbc, cutoff);
        std::cerr << "Failed to catch flat cell." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}

    // 6. Positions outside cell (PBC)
    // Atom at 1.5 in box of size 1.0. Should wrap to 0.5.
    {
        std::vector<double> out_pos = {1.5, 0.5, 0.5,  0.5, 0.5, 0.5}; // 0 wraps to 0.5. 1 is at 0.5.
        // They should be on top of each other -> neighbors.
        nl.build(out_pos, cell, pbc, cutoff);
        assert(nl.neighbors[0].size() == 1);
        assert(nl.neighbors[0][0].index == 1);
    }
    
    // 7. Positions outside cell (Non-PBC)
    // Atom at 100.0. Box 1.0. Cutoff 1.0.
    // Should be far away.
    {
        std::vector<double> far_pos = {0,0,0, 100,0,0};
        std::vector<bool> no_pbc = {false, false, false};
        nl.build(far_pos, cell, no_pbc, cutoff);
        assert(nl.neighbors[0].empty());
    }

    // 8. Extreme Cutoffs
    // Small cutoff
    {
        std::vector<double> pair = {0,0,0, 0.5,0,0};
        nl.build(pair, cell, pbc, 0.1); // Too small
        assert(nl.neighbors[0].empty());
    }
    // Large cutoff
    {
        std::vector<double> pair = {0,0,0, 0.5,0,0};
        std::vector<bool> no_pbc = {false, false, false};
        nl.build(pair, cell, no_pbc, 100.0); // Huge
        assert(nl.neighbors[0].size() == 1);
    }

    std::cout << "Passed." << std::endl;
}

// Reference list, by trying every image within reach.
static std::vector<std::array<int, 4>> brute_force(
    const std::vector<double>& pos, const std::vector<double>& cell,
    const std::vector<bool>& pbc, double cutoff, int reach = 6) {
    const int n = static_cast<int>(pos.size()) / 3;
    std::vector<std::array<int, 4>> out;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int rx = pbc[0] ? -reach : 0; rx <= (pbc[0] ? reach : 0); ++rx) {
                for (int ry = pbc[1] ? -reach : 0; ry <= (pbc[1] ? reach : 0); ++ry) {
                    for (int rz = pbc[2] ? -reach : 0; rz <= (pbc[2] ? reach : 0); ++rz) {
                        if (i == j && rx == 0 && ry == 0 && rz == 0) continue;
                        double d2 = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            const double d = pos[3 * j + k] - pos[3 * i + k] +
                                             rx * cell[k] + ry * cell[3 + k] + rz * cell[6 + k];
                            d2 += d * d;
                        }
                        if (d2 < cutoff * cutoff) out.push_back({i, j, rx, ry});
                    }
                }
            }
        }
    }
    return out;
}

static bool has_neighbor(const NeighborList& nl, int i, int j, int rx, int ry) {
    for (const auto& n : nl.neighbors[i]) {
        if (n.index == j && n.rx == rx && n.ry == ry) return true;
    }
    return false;
}

// A wrapped fractional coordinate has to land in the bin that contains it.
//
// `s - floor(s)` is not guaranteed to be < 1: for an s that is a tiny negative
// number the subtraction rounds up to exactly 1.0. Binning that by `% nbins`
// used to put the atom at the BOTTOM of the cell while its wrapped position sat
// at the TOP, so the fixed-radius bin search looked a whole lattice vector away
// and silently dropped its bonds.
//
// Graphene's conventional basis puts the second carbon at fractional
// (-1/3, 2/3), which is exactly the case that rounds; a supercell of it lost
// bonds and |E|max came out below the analytic 3|t|. Every supercell here must
// give a perfect honeycomb: three neighbours per atom, no more and no fewer.
void test_wrapped_boundary_coordinates() {
    std::cout << "Testing atoms on a wrapped cell boundary..." << std::endl;
    const double a_cc = 1.42;
    const double a = std::sqrt(3.0) * a_cc;

    // The primitive cell, and the conventional two-atom basis: the second
    // carbon at fractional (-1/3, 2/3), i.e. outside the cell.
    const std::vector<double> primitive = {a, 0, 0, 0.5 * a, 1.5 * a_cc, 0, 0, 0, 20.0};

    for (int sc = 1; sc <= 6; ++sc) {
        std::vector<double> cell = {primitive[0] * sc, primitive[1] * sc, primitive[2] * sc,
                                    primitive[3] * sc, primitive[4] * sc, primitive[5] * sc,
                                    primitive[6],      primitive[7],      primitive[8]};
        std::vector<double> pos;
        for (int i = 0; i < sc; ++i) {
            for (int j = 0; j < sc; ++j) {
                // Offsets accumulated from the cell rows, the way a caller
                // tiling a cell does; this is what lands a fractional
                // coordinate on -0 or on a tiny negative number.
                const double ox = i * primitive[0] + j * primitive[3];
                const double oy = i * primitive[1] + j * primitive[4];
                pos.push_back(ox);       pos.push_back(oy);          pos.push_back(10.0);
                pos.push_back(ox);       pos.push_back(oy + a_cc);   pos.push_back(10.0);
            }
        }

        const std::vector<bool> pbc = {true, true, false};
        NeighborList nl;
        nl.build(pos, cell, pbc, 1.6);

        const int n = static_cast<int>(pos.size()) / 3;
        for (int i = 0; i < n; ++i) {
            assert(nl.neighbors[i].size() == 3 &&
                   "every carbon has exactly three neighbours at 1.42 A");
        }
        for (const auto& e : brute_force(pos, cell, pbc, 1.6)) {
            assert(has_neighbor(nl, e[0], e[1], e[2], e[3]) &&
                   "binned search must find every bond brute force finds");
        }
    }
    std::cout << "Passed." << std::endl;
}

// The regression test for the wrap above, with a deterministic trigger.
//
// An atom at a tiny NEGATIVE fractional coordinate is the exact case: floor is
// -1, and `s - (-1)` rounds to exactly 1.0 because |s| is below the double
// epsilon. The atom then belongs at the far face of the cell, and `% nbins`
// used to bin it at the near face instead -- a whole cell away from where its
// wrapped position sits -- so the fixed-radius bin search never looked where
// its neighbour actually was.
//
// Here atom 0 sits a hair below fractional 0 and atom 1 just inside the far
// face, 0.5 apart across the periodic boundary. Before the fix atom 0 found no
// neighbour at all.
void test_tiny_negative_fractional_coordinate() {
    std::cout << "Testing a tiny negative fractional coordinate..." << std::endl;
    const std::vector<double> cell = {6.0, 0, 0, 0, 6.0, 0, 0, 0, 6.0};
    const std::vector<bool> pbc = {true, true, true};
    const std::vector<double> pos = {-1e-18, 3.0, 3.0,
                                      5.5,   3.0, 3.0};

    NeighborList nl;
    nl.build(pos, cell, pbc, 1.0);

    assert(nl.neighbors[0].size() == 1 && "the bond across the boundary must be found");
    assert(nl.neighbors[1].size() == 1);
    assert(has_neighbor(nl, 0, 1, -1, 0));
    assert(has_neighbor(nl, 1, 0, 1, 0));

    // Same geometry, written with the atom a hair ABOVE zero instead: the
    // coordinates differ by 2e-18, the bonds must not differ at all.
    const std::vector<double> mirrored = {1e-18, 3.0, 3.0, 5.5, 3.0, 3.0};
    NeighborList mirrored_nl;
    mirrored_nl.build(mirrored, cell, pbc, 1.0);
    assert(mirrored_nl.neighbors[0].size() == nl.neighbors[0].size() &&
           "a 2e-18 change in a coordinate must not change the bond list");

    // And with the atom a whole cell out, which wraps by a full lattice vector.
    const std::vector<double> outside = {-6.0 - 1e-18, 3.0, 3.0, 5.5, 3.0, 3.0};
    NeighborList outside_nl;
    outside_nl.build(outside, cell, pbc, 1.0);
    assert(outside_nl.neighbors[0].size() == 1);
    assert(has_neighbor(outside_nl, 0, 1, -2, 0) &&
           "shifts are reported against the coordinates as given");
    std::cout << "Passed." << std::endl;
}

// Randomized comparison against brute force.
//
// The binned search is an optimization of "try every image", so the two must
// agree exactly -- no bond found by one and missed by the other. The cases are
// drawn to cover what tripped the wrap above: strongly skewed cells, atoms
// exactly on a face, atoms whole cells outside the box, and mixed periodicity.
void test_matches_brute_force_randomized() {
    std::cout << "Testing against brute force over randomized cells..." << std::endl;
    std::mt19937 rng(12345);  // fixed seed: a failure here must be reproducible
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int checked = 0;

    for (int trial = 0; trial < 1000; ++trial) {
        const int n = 1 + static_cast<int>(u(rng) * 8);
        const double L = 2.0 + 6.0 * u(rng);
        std::vector<double> cell = {L, 0, 0,
                                    (u(rng) - 0.5) * 1.6 * L, L * (0.5 + u(rng)), 0,
                                    (u(rng) - 0.5) * 1.6 * L, (u(rng) - 0.5) * L,
                                    L * (0.5 + u(rng))};
        const std::vector<bool> pbc = {u(rng) < 0.85, u(rng) < 0.85, u(rng) < 0.85};
        const double cutoff = 0.3 + 3.0 * u(rng);

        std::vector<double> pos;
        for (int i = 0; i < n; ++i) {
            double s[3];
            for (int k = 0; k < 3; ++k) {
                const double pick = u(rng);
                if (pick < 0.20) s[k] = 0.0;                              // on a face
                else if (pick < 0.30) s[k] = 1.0;                         // on the far face
                else if (pick < 0.45) s[k] = std::floor(u(rng) * 5) - 2;  // whole cells out
                else s[k] = (u(rng) - 0.5) * 3.0;                         // anywhere
            }
            for (int k = 0; k < 3; ++k) {
                pos.push_back(s[0] * cell[k] + s[1] * cell[3 + k] + s[2] * cell[6 + k]);
            }
        }

        NeighborList nl;
        try {
            nl.build(pos, cell, pbc, cutoff);
        } catch (const std::invalid_argument&) {
            continue;  // a degenerate cell is rejected, which is its own contract
        }

        for (const auto& e : brute_force(pos, cell, pbc, cutoff)) {
            if (!has_neighbor(nl, e[0], e[1], e[2], e[3])) {
                std::cerr << "trial " << trial << ": binned search missed bond "
                          << e[0] << "->" << e[1] << " R=(" << e[2] << "," << e[3]
                          << ") with cutoff " << cutoff << std::endl;
                exit(1);
            }
        }
        ++checked;
    }
    std::cout << "Passed (" << checked << " random cells)." << std::endl;
}

int main() {
    test_simple_cubic_pbc();
    test_non_pbc();
    test_large_system();
    test_mixed_pbc();
    test_exact_indices();
    test_robustness();
    test_wrapped_boundary_coordinates();
    test_tiny_negative_fractional_coordinate();
    test_matches_brute_force_randomized();
    return 0;
}
