#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace vbcsr {
namespace atomic {

class NeighborList {
public:
    struct Neighbor {
        int index;
        int rx, ry, rz;
    };

    // Neighbors for each atom: neighbors[i] contains list of Neighbor structs.
    //
    // Shift vectors are relative to the POSITIONS AS GIVEN. Wrapping into the
    // cell happens internally, for binning only; the lattice translation each
    // atom picked up there is undone before a neighbour is recorded, so a
    // caller reconstructs a bond as r_j + R - r_i using its own coordinates and
    // never has to know whether they were inside the cell.
    std::vector<std::vector<Neighbor>> neighbors;

    NeighborList() = default;

    // Build the neighbor list
    // positions: Nx3 vector
    // cell: 3x3 matrix (row-major: ax, ay, az, bx, by, bz, cx, cy, cz)
    // pbc: 3 bools
    // cutoff: search radius
    void build(const std::vector<double>& positions, 
               const std::vector<double>& cell, 
               const std::vector<bool>& pbc, 
               double cutoff) {
        
        // Input Validation
        if (positions.size() % 3 != 0) {
            throw std::invalid_argument("Positions vector size must be a multiple of 3.");
        }
        if (cell.size() != 9) {
            throw std::invalid_argument("Cell vector size must be 9.");
        }
        if (pbc.size() != 3) {
            throw std::invalid_argument("PBC vector size must be 3.");
        }
        if (cutoff <= 0.0) {
            // If cutoff is zero or negative, no neighbors can be found.
            // Just return empty lists.
            int n_atoms = positions.size() / 3;
            neighbors.assign(n_atoms, std::vector<Neighbor>());
            return;
        }

        int n_atoms = positions.size() / 3;
        neighbors.assign(n_atoms, std::vector<Neighbor>());

        if (n_atoms == 0) return;

        // 1. Compute Reciprocal Cell and Face Distances
        // Cell vectors
        std::array<double, 3> a = {cell[0], cell[1], cell[2]};
        std::array<double, 3> b = {cell[3], cell[4], cell[5]};
        std::array<double, 3> c = {cell[6], cell[7], cell[8]};

        // Cross products
        auto cross = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
            return std::array<double, 3>{
                u[1]*v[2] - u[2]*v[1],
                u[2]*v[0] - u[0]*v[2],
                u[0]*v[1] - u[1]*v[0]
            };
        };

        auto dot = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
            return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
        };

        auto norm = [](const std::array<double, 3>& u) {
            return std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
        };

        std::array<double, 3> bxc = cross(b, c);
        std::array<double, 3> cxa = cross(c, a);
        std::array<double, 3> axb = cross(a, b);

        // Signed triple product (cell determinant). Fractional coordinates must
        // divide by the SIGNED determinant: for a left-handed cell-vector
        // ordering (a.(bxc) < 0) the cross products bxc/cxa/axb are anti-parallel
        // to the reciprocal directions, so dividing by |det| would negate the
        // fractional coordinates and wrap atoms to the wrong image. Face
        // distances (heights) use the absolute volume.
        double det = dot(a, bxc);
        double vol = std::abs(det);

        if (vol < 1e-8) {
            throw std::invalid_argument("Cell volume is too small or negative.");
        }
        
        // Face distances (heights of the parallelepiped)
        // h_a = vol / |b x c|
        // h_b = vol / |c x a|
        // h_c = vol / |a x b|
        double h_a = vol / norm(bxc);
        double h_b = vol / norm(cxa);
        double h_c = vol / norm(axb);

        std::array<double, 3> face_dist = {h_a, h_b, h_c};

        // 2. Determine Bin Size and Number of Bins
        // Bin size = cutoff, so a fixed neighbourhood of bins suffices; the
        // search range below still widens when the cell forces smaller bins.
        double bin_size = cutoff;
        if (bin_size < 1e-5) bin_size = 1.0; // Safety

        std::array<int, 3> nbins_c;
        for(int i=0; i<3; ++i) {
            nbins_c[i] = static_cast<int>(face_dist[i] / bin_size);
            if (nbins_c[i] < 1) nbins_c[i] = 1;
        }

        // 3. Sort Atoms into Bins
        // Scaled (fractional) coordinates map atoms to bins: s = inv_cell * r,
        // with inv_cell = (1/det) * [bxc; cxa; axb]^T.
        // Inverse cell rows, applied below via manual dot products:
        // Row 0 of inv: bxc / vol
        // Row 1 of inv: cxa / vol
        // Row 2 of inv: axb / vol
        
        std::vector<int> atom_to_bin(n_atoms);
        std::vector<std::vector<int>> bins(nbins_c[0] * nbins_c[1] * nbins_c[2]);
        
        // Create working positions (wrapped if PBC)
        std::vector<double> working_positions = positions;

        // The integer lattice translation wrapping applied to each atom:
        // r_working = r_input + wrap_shift * cell. Recorded so the shifts
        // reported to the caller can be expressed against the input
        // coordinates, whether or not those lay inside the cell.
        std::vector<std::array<int, 3>> wrap_shift(n_atoms, {0, 0, 0});

        for(int i=0; i<n_atoms; ++i) {
            double rx = positions[3*i];
            double ry = positions[3*i+1];
            double rz = positions[3*i+2];

            // Scaled coordinates s = inv_cell * r
            double sx = dot({rx, ry, rz}, bxc) / det;
            double sy = dot({rx, ry, rz}, cxa) / det;
            double sz = dot({rx, ry, rz}, axb) / det;

            // Wrap scaled coordinates if PBC, remembering by how much: an atom
            // at fractional -1/3 moves to 2/3 and gains one lattice vector.
            // Shift vectors are ints, so a coordinate absurdly far outside the
            // cell is rejected rather than silently wrapped into a plausible
            // but wrong image.
            auto wrap = [](double& val) {
                double cells = std::floor(val);
                if (!(std::abs(cells) < 1e6)) {
                    throw std::invalid_argument(
                        "Atom position is more than 1e6 cells outside the unit cell, or is "
                        "not finite; its periodic image cannot be represented.");
                }
                val -= cells;
                // `val - floor(val)` is not guaranteed to land in [0, 1): for a
                // val that is a tiny negative number the subtraction rounds up
                // to exactly 1.0. Left there, the atom is binned at the BOTTOM
                // of the cell (see the clamp below) while its wrapped position
                // sits at the TOP, so the one-bin search looks a whole lattice
                // vector away from its real neighbours and silently drops the
                // bonds. Fold once more so position and bin always agree.
                if (val >= 1.0) {
                    val = 0.0;
                    cells += 1.0;
                }
                return -static_cast<int>(cells);
            };

            bool modified = false;
            if (pbc[0]) { wrap_shift[i][0] = wrap(sx); modified = true; }
            if (pbc[1]) { wrap_shift[i][1] = wrap(sy); modified = true; }
            if (pbc[2]) { wrap_shift[i][2] = wrap(sz); modified = true; }

            if (modified) {
                // Update working_positions
                // r = s * cell
                // r = sx * a + sy * b + sz * c
                double new_rx = sx * a[0] + sy * b[0] + sz * c[0];
                double new_ry = sx * a[1] + sy * b[1] + sz * c[1];
                double new_rz = sx * a[2] + sy * b[2] + sz * c[2];
                
                working_positions[3*i] = new_rx;
                working_positions[3*i+1] = new_ry;
                working_positions[3*i+2] = new_rz;
            }

            // Bin indices.
            //
            // A bin must be the one that CONTAINS the atom's working position:
            // the search below scans a fixed neighbourhood of bins around it,
            // so a bin that disagrees with the position by even one slab makes
            // the search look in the wrong place. The out-of-range cases here
            // are pure floating-point residue -- sx is already in [0, 1), but
            // sx * nbins can still round to exactly nbins -- so they are
            // CLAMPED to the nearest bin, never wrapped to the far side.
            auto bin_of = [](double scaled, int nbins) {
                int index = static_cast<int>(std::floor(scaled * nbins));
                if (index < 0) index = 0;
                if (index >= nbins) index = nbins - 1;
                return index;
            };

            const int bx = bin_of(sx, nbins_c[0]);
            const int by = bin_of(sy, nbins_c[1]);
            const int bz = bin_of(sz, nbins_c[2]);

            int bin_idx = bx + nbins_c[0] * (by + nbins_c[1] * bz);
            bins[bin_idx].push_back(i);
        }

        // 4. Neighbor Search
        // For each bin, scan the neighbouring bins (including self). With
        // bin_size >= cutoff one layer per direction suffices; when the cell
        // yields smaller physical bins, the range grows to ceil(cutoff/bin).
        int search_range_x = 1;
        int search_range_y = 1;
        int search_range_z = 1;

        // Calculate search range based on cutoff and bin physical size
        // bin_phys_size approx face_dist / nbins
        // We need search_range * bin_phys_size >= cutoff
        // So search_range >= cutoff / (face_dist / nbins) = (cutoff * nbins) / face_dist
        
        auto get_search_range = [&](double dist, int nbins) {
            double bin_phys = dist / nbins;
            // We add a small buffer or use ceil
            int range = static_cast<int>(std::ceil(cutoff / bin_phys));
            if (range < 1) range = 1;
            return range;
        };

        search_range_x = get_search_range(face_dist[0], nbins_c[0]);
        search_range_y = get_search_range(face_dist[1], nbins_c[1]);
        search_range_z = get_search_range(face_dist[2], nbins_c[2]);

        double cutoff_sq = cutoff * cutoff;

        for(int bz = 0; bz < nbins_c[2]; ++bz) {
            for(int by = 0; by < nbins_c[1]; ++by) {
                for(int bx = 0; bx < nbins_c[0]; ++bx) {
                    
                    int bin_idx = bx + nbins_c[0] * (by + nbins_c[1] * bz);
                    const auto& atoms_in_current_bin = bins[bin_idx];
                    
                    if (atoms_in_current_bin.empty()) continue;

                    // Iterate over neighbor bins
                    for(int dz = -search_range_z; dz <= search_range_z; ++dz) {
                        for(int dy = -search_range_y; dy <= search_range_y; ++dy) {
                            for(int dx = -search_range_x; dx <= search_range_x; ++dx) {
                                
                                // Neighbor bin indices
                                int nbx = bx + dx;
                                int nby = by + dy;
                                int nbz = bz + dz;
                                
                                // Shift vector (for PBC)
                                double shift_x = 0.0;
                                double shift_y = 0.0;
                                double shift_z = 0.0;
                                
                                // Handle PBC
                                if (pbc[0]) {
                                    shift_x = std::floor(static_cast<double>(nbx) / nbins_c[0]);
                                    nbx = nbx % nbins_c[0];
                                    if (nbx < 0) nbx += nbins_c[0];
                                } else {
                                    if (nbx < 0 || nbx >= nbins_c[0]) continue;
                                }
                                
                                if (pbc[1]) {
                                    shift_y = std::floor(static_cast<double>(nby) / nbins_c[1]);
                                    nby = nby % nbins_c[1];
                                    if (nby < 0) nby += nbins_c[1];
                                } else {
                                    if (nby < 0 || nby >= nbins_c[1]) continue;
                                }
                                
                                if (pbc[2]) {
                                    shift_z = std::floor(static_cast<double>(nbz) / nbins_c[2]);
                                    nbz = nbz % nbins_c[2];
                                    if (nbz < 0) nbz += nbins_c[2];
                                } else {
                                    if (nbz < 0 || nbz >= nbins_c[2]) continue;
                                }
                                
                                int nbin_idx = nbx + nbins_c[0] * (nby + nbins_c[1] * nbz);
                                const auto& atoms_in_neighbor_bin = bins[nbin_idx];
                                
                                if (atoms_in_neighbor_bin.empty()) continue;
                                
                                // Cartesian shift
                                double Rx = shift_x * a[0] + shift_y * b[0] + shift_z * c[0];
                                double Ry = shift_x * a[1] + shift_y * b[1] + shift_z * c[1];
                                double Rz = shift_x * a[2] + shift_y * b[2] + shift_z * c[2];

                                // Check pairs
                                for(int i : atoms_in_current_bin) {
                                    double ix = working_positions[3*i];
                                    double iy = working_positions[3*i+1];
                                    double iz = working_positions[3*i+2];
                                    
                                    for(int j : atoms_in_neighbor_bin) {
                                        // Full lists: every ordered (i, j) pair is
                                        // recorded; only the self-pair in the home
                                        // image is skipped.
                                        if (i == j && shift_x == 0 && shift_y == 0 && shift_z == 0) continue;
                                        
                                        // Distance vector r_ji = r_j + R - r_i
                                        double dx_ij = working_positions[3*j] + Rx - ix;
                                        double dy_ij = working_positions[3*j+1] + Ry - iy;
                                        double dz_ij = working_positions[3*j+2] + Rz - iz;
                                        
                                        double r2 = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;

                                        if (r2 < cutoff_sq) {
                                            // Back to the caller's coordinates:
                                            //   r_j^wrap + R = r_j + (W_j + R)
                                            //   r_i^wrap     = r_i + W_i
                                            // so the shift against the input
                                            // positions is R + W_j - W_i. For a
                                            // structure already inside the cell
                                            // every W is zero and this is R.
                                            neighbors[i].push_back({
                                                j,
                                                (int)shift_x + wrap_shift[j][0] - wrap_shift[i][0],
                                                (int)shift_y + wrap_shift[j][1] - wrap_shift[i][1],
                                                (int)shift_z + wrap_shift[j][2] - wrap_shift[i][2]});
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Helper functions
    const std::vector<Neighbor>& get_neighbors(int i) const {
        if (i < 0 || i >= neighbors.size()) {
            throw std::out_of_range("Atom index out of range");
        }
        return neighbors[i];
    }

    std::vector<std::pair<int, int>> get_pairs_with_shift(int rx, int ry, int rz) const {
        std::vector<std::pair<int, int>> pairs;
        for(size_t i=0; i<neighbors.size(); ++i) {
            for(const auto& n : neighbors[i]) {
                if (n.rx == rx && n.ry == ry && n.rz == rz) {
                    pairs.push_back({(int)i, n.index});
                }
            }
        }
        return pairs;
    }
};

} // namespace atomic
} // namespace vbcsr
