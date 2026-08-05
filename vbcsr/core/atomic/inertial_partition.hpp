#ifndef VBCSR_CORE_ATOMIC_INERTIAL_PARTITION_HPP
#define VBCSR_CORE_ATOMIC_INERTIAL_PARTITION_HPP

// Recursive Inertial Bisection over weighted points, serial and distributed.
//
// Recursively split the points perpendicular to their principal axis of
// inertia -- the max-spread direction -- at the weighted median. The cut
// planes tilt to follow the point distribution, so each region is compact
// (small cut surface, hence small halo) without being constrained to a box.
//
// The tree records every cut plane, so an arbitrary point classifies against
// the same planes the points were bisected by. That binding is the reason the
// tree is returned at all: a consumer that has to place other objects -- grid
// points, field samples -- in the same regions gets `owner(point)` for free,
// with no communication and no second partitioning scheme to keep in sync.
//
// Purely geometric: positions, weights and a part count. No knowledge of what
// the points are or what the weights mean; a caller that wants to balance
// orbital counts rather than atoms passes orbital counts as weights.
//
// Alternatives considered, and why not: a Morton/Hilbert chunk has locality
// but a jagged boundary; an axis-aligned bisection is rectangular but cannot
// tilt to fit the geometry; a multilevel graph partitioner cuts communication
// volume marginally better but needs the graph itself -- which for a large
// system is exactly the object we are trying to avoid materializing.

#include "../detail/kernels/lapack_api.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>

namespace vbcsr {
namespace atomic {

/// A node of the bisection tree: a leaf owning a rank, or a cut plane.
///
/// A point p goes left iff dot(p, axis) < threshold -- the same test the atoms
/// were split by, which is what binds point ownership to atom ownership.
struct InertialCut {
    bool leaf = true;
    int rank = 0;
    double axis[3] = {0.0, 0.0, 0.0};
    double threshold = 0.0;
    std::unique_ptr<InertialCut> left;
    std::unique_ptr<InertialCut> right;
};

/// Result of partitioning atoms: ownership plus the tree that reproduces it.
struct InertialPartition {
    std::vector<int> owner;            ///< Owning rank of each input atom.
    std::vector<int> counts;           ///< Atoms per rank.
    std::unique_ptr<InertialCut> tree; ///< Cut planes, for classifying points.
};

namespace detail {

/// Principal axis of a weighted point set: the eigenvector of the inertia
/// tensor with the largest eigenvalue, i.e. the direction of maximum spread.
///
/// The sign is canonicalized so the partition is reproducible run to run --
/// LAPACK is free to return either sign, and while a flip only swaps the two
/// children, it would make the rank assignment non-deterministic.
inline void PrincipalAxis(const double* inertia, double* axis_out) {
    using vbcsr::vbcsr_lapack_int;
    double a[9];
    std::copy(inertia, inertia + 9, a);

    const vbcsr_lapack_int n = 3;
    const vbcsr_lapack_int lwork = 32;
    double work[32];
    double evals[3];
    vbcsr_lapack_int info = 0;
    vbcsr::dsyev_("V", "U", &n, a, &n, evals, work, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("Inertial partition: 3x3 eigenproblem failed.");
    }

    // dsyev returns ascending eigenvalues with eigenvectors in columns; the
    // last column is the principal axis.
    for (int i = 0; i < 3; ++i) axis_out[i] = a[6 + i];

    int dominant = 0;
    for (int i = 1; i < 3; ++i) {
        if (std::abs(axis_out[i]) > std::abs(axis_out[dominant])) dominant = i;
    }
    if (axis_out[dominant] < 0.0) {
        for (int i = 0; i < 3; ++i) axis_out[i] = -axis_out[i];
    }
}

}  // namespace detail

/// Partitions weighted points into `size` compact, work-balanced regions.
///
/// Args:
///   positions: (n, 3) Cartesian coordinates, row-major.
///   weights: Per-point work, e.g. orbital count. Empty means uniform.
///   size: Number of parts.
/// Returns:
///   Ownership, per-rank counts, and the cut tree.
inline InertialPartition PartitionByInertia(const std::vector<double>& positions,
                                            const std::vector<double>& weights,
                                            int size) {
    if (positions.size() % 3 != 0) {
        throw std::runtime_error("Inertial partition needs an (n, 3) position array.");
    }
    const int n = static_cast<int>(positions.size() / 3);
    if (size < 1) {
        throw std::runtime_error("Inertial partition needs at least one part.");
    }

    std::vector<double> w(n, 1.0);
    if (!weights.empty()) {
        if (static_cast<int>(weights.size()) != n) {
            throw std::runtime_error("Inertial partition: one weight per position.");
        }
        w = weights;
    }

    InertialPartition result;
    result.owner.assign(n, 0);
    result.counts.assign(size, 0);

    // `indices` is the atom subset of the current node; `lo`/`hi` the rank
    // range it must be split across.
    std::function<std::unique_ptr<InertialCut>(std::vector<int>, int, int)> build =
        [&](std::vector<int> indices, int lo, int hi) -> std::unique_ptr<InertialCut> {
        auto node = std::make_unique<InertialCut>();
        const int n_parts = hi - lo;
        if (n_parts <= 1 || indices.size() <= 1) {
            node->leaf = true;
            node->rank = lo;
            for (int index : indices) result.owner[index] = lo;
            return node;
        }

        double weight_sum = 0.0;
        double centroid[3] = {0.0, 0.0, 0.0};
        for (int index : indices) {
            weight_sum += w[index];
            for (int x = 0; x < 3; ++x) centroid[x] += w[index] * positions[3 * index + x];
        }
        if (weight_sum <= 0.0) weight_sum = 1.0;
        for (int x = 0; x < 3; ++x) centroid[x] /= weight_sum;

        double inertia[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int index : indices) {
            double d[3];
            for (int x = 0; x < 3; ++x) d[x] = positions[3 * index + x] - centroid[x];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) inertia[3 * a + b] += w[index] * d[a] * d[b];
            }
        }
        detail::PrincipalAxis(inertia, node->axis);

        std::vector<double> projection(indices.size(), 0.0);
        for (size_t i = 0; i < indices.size(); ++i) {
            const int index = indices[i];
            double value = 0.0;
            for (int x = 0; x < 3; ++x) {
                value += (positions[3 * index + x] - centroid[x]) * node->axis[x];
            }
            projection[i] = value;
        }

        std::vector<int> order(indices.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return projection[a] < projection[b]; });

        // Weighted median at the proportional rank split, so a node covering
        // an odd rank count still balances work rather than atom count.
        const int left_parts = n_parts / 2;
        const double target = weight_sum * left_parts / n_parts;
        double running = 0.0;
        size_t split = 0;
        while (split < order.size() && running < target) {
            running += w[indices[order[split]]];
            ++split;
        }
        split = std::min(std::max<size_t>(split, 1), indices.size() - 1);

        // The plane sits midway between the two atoms that straddle the split,
        // in absolute coordinates, so ClassifyPoints can compare dot(p, axis)
        // directly against it.
        double centroid_projection = 0.0;
        for (int x = 0; x < 3; ++x) centroid_projection += centroid[x] * node->axis[x];
        node->threshold = centroid_projection +
                          0.5 * (projection[order[split - 1]] + projection[order[split]]);

        std::vector<int> left_indices, right_indices;
        left_indices.reserve(split);
        right_indices.reserve(indices.size() - split);
        for (size_t i = 0; i < order.size(); ++i) {
            const int index = indices[order[i]];
            if (i < split) {
                left_indices.push_back(index);
            } else {
                right_indices.push_back(index);
            }
        }

        node->leaf = false;
        node->left = build(std::move(left_indices), lo, lo + left_parts);
        node->right = build(std::move(right_indices), lo + left_parts, hi);
        return node;
    };

    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    if (size > 1 && n > 0) {
        result.tree = build(std::move(all), 0, size);
    } else {
        result.tree = std::make_unique<InertialCut>();
        result.tree->leaf = true;
        result.tree->rank = 0;
    }

    for (int index = 0; index < n; ++index) result.counts[result.owner[index]] += 1;
    return result;
}

/// Owner of a single point under an existing cut tree.
/// Distributed recursive inertial bisection.
///
/// The same partition PartitionByInertia produces, computed without any rank
/// holding every position. Each rank contributes whatever points it happens to
/// hold -- a block split of an input file is the usual source -- and receives
/// the whole cut tree plus the destination part of each of its own points.
///
/// The tree is built level by level, and every rank runs the identical
/// arithmetic on identical MPI_Allreduce results, so all ranks end up with the
/// same tree. Per level the cost is a handful of small collectives: one to
/// accumulate every node's mass, centroid and second moments, `median_steps`
/// to bisect for the cut, and one to centre the cut in the gap it landed in.
/// Total O(log P) rounds of O(P) doubles -- kilobytes, against the gigabytes a
/// gathered point set would cost.
///
/// Args:
///   local_positions: (n_local, 3) row-major, this rank's points only.
///   local_weights: Per-point work, empty for uniform.
///   size: Number of parts (normally the communicator size).
///   comm: Ranks holding the points. MPI_COMM_NULL runs it serially.
///   median_steps: Bisection steps per cut. The result is then snapped to the
///     straddling gap, so this only has to localize the cut to one gap.
/// Returns:
///   `owner` sized to the local point count; `counts` global per part; `tree`
///   identical on every rank.
inline InertialPartition PartitionByInertiaDistributed(
    const std::vector<double>& local_positions,
    const std::vector<double>& local_weights,
    int size,
    MPI_Comm comm,
    int median_steps = 40) {
    if (local_positions.size() % 3 != 0) {
        throw std::runtime_error("Inertial partition needs an (n, 3) position array.");
    }
    if (size < 1) {
        throw std::runtime_error("Inertial partition needs at least one part.");
    }
    const int n_local = static_cast<int>(local_positions.size() / 3);

    std::vector<double> w(n_local, 1.0);
    if (!local_weights.empty()) {
        if (static_cast<int>(local_weights.size()) != n_local) {
            throw std::runtime_error("Inertial partition: one weight per position.");
        }
        w = local_weights;
    }

    InertialPartition result;
    result.owner.assign(n_local, 0);
    result.counts.assign(size, 0);

    // node_of[i] indexes the current level's node list for local point i.
    std::vector<int> node_of(n_local, 0);

    struct Node {
        int lo = 0;
        int hi = 0;  // exclusive
        InertialCut* slot = nullptr;
    };
    result.tree.reset(new InertialCut());
    std::vector<Node> level{Node{0, size, result.tree.get()}};

    const auto all_reduce = [&](double* data, int count, MPI_Op op) {
        if (comm != MPI_COMM_NULL) {
            MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, op, comm);
        }
    };

    while (!level.empty()) {
        const int n_node = static_cast<int>(level.size());

        // --- mass, centroid and second moments of every node, in one call ---
        constexpr int kStats = 10;
        std::vector<double> stats(static_cast<size_t>(n_node) * kStats, 0.0);
        for (int i = 0; i < n_local; ++i) {
            const int nd = node_of[i];
            if (nd < 0) continue;
            const double* p = &local_positions[3 * static_cast<size_t>(i)];
            double* a = &stats[static_cast<size_t>(nd) * kStats];
            a[0] += w[i];
            for (int k = 0; k < 3; ++k) a[1 + k] += w[i] * p[k];
            a[4] += w[i] * p[0] * p[0];
            a[5] += w[i] * p[1] * p[1];
            a[6] += w[i] * p[2] * p[2];
            a[7] += w[i] * p[0] * p[1];
            a[8] += w[i] * p[0] * p[2];
            a[9] += w[i] * p[1] * p[2];
        }
        all_reduce(stats.data(), static_cast<int>(stats.size()), MPI_SUM);

        std::vector<double> axes(static_cast<size_t>(n_node) * 3, 0.0);
        std::vector<char> splitting(n_node, 0);
        for (int nd = 0; nd < n_node; ++nd) {
            const Node& node = level[static_cast<size_t>(nd)];
            if (node.hi - node.lo <= 1) continue;
            const double* a = &stats[static_cast<size_t>(nd) * kStats];
            const double mass = a[0];
            if (mass <= 0.0) continue;
            const double cx = a[1] / mass, cy = a[2] / mass, cz = a[3] / mass;
            // Covariance about the node's own centroid.
            const double cov[9] = {
                a[4] / mass - cx * cx, a[7] / mass - cx * cy, a[8] / mass - cx * cz,
                a[7] / mass - cx * cy, a[5] / mass - cy * cy, a[9] / mass - cy * cz,
                a[8] / mass - cx * cz, a[9] / mass - cy * cz, a[6] / mass - cz * cz};
            detail::PrincipalAxis(cov, &axes[static_cast<size_t>(nd) * 3]);
            splitting[nd] = 1;
        }

        const auto projection = [&](int i, int nd) {
            const double* p = &local_positions[3 * static_cast<size_t>(i)];
            const double* ax = &axes[static_cast<size_t>(nd) * 3];
            return p[0] * ax[0] + p[1] * ax[1] + p[2] * ax[2];
        };

        // --- bracket each node's projection range ---
        std::vector<double> bracket(static_cast<size_t>(n_node) * 2,
                                    std::numeric_limits<double>::lowest());
        for (int i = 0; i < n_local; ++i) {
            const int nd = node_of[i];
            if (nd < 0 || !splitting[nd]) continue;
            const double proj = projection(i, nd);
            double& neg_min = bracket[2 * static_cast<size_t>(nd)];
            double& max = bracket[2 * static_cast<size_t>(nd) + 1];
            neg_min = std::max(neg_min, -proj);
            max = std::max(max, proj);
        }
        all_reduce(bracket.data(), static_cast<int>(bracket.size()), MPI_MAX);

        std::vector<double> lo_bound(n_node, 0.0), hi_bound(n_node, 0.0), target(n_node, 0.0);
        for (int nd = 0; nd < n_node; ++nd) {
            lo_bound[nd] = -bracket[2 * static_cast<size_t>(nd)];
            hi_bound[nd] = bracket[2 * static_cast<size_t>(nd) + 1];
            if (!splitting[nd]) continue;
            // Balance work, not point count: an odd part range still splits by
            // the fraction of parts on each side.
            const Node& node = level[static_cast<size_t>(nd)];
            const int mid_rank = node.lo + (node.hi - node.lo) / 2;
            target[nd] = stats[static_cast<size_t>(nd) * kStats] *
                         static_cast<double>(mid_rank - node.lo) /
                         static_cast<double>(node.hi - node.lo);
        }

        // --- bisect for the weight-balanced cut ---
        std::vector<double> left_mass(n_node, 0.0);
        for (int step = 0; step < median_steps; ++step) {
            std::fill(left_mass.begin(), left_mass.end(), 0.0);
            for (int i = 0; i < n_local; ++i) {
                const int nd = node_of[i];
                if (nd < 0 || !splitting[nd]) continue;
                if (projection(i, nd) < 0.5 * (lo_bound[nd] + hi_bound[nd])) left_mass[nd] += w[i];
            }
            all_reduce(left_mass.data(), n_node, MPI_SUM);
            for (int nd = 0; nd < n_node; ++nd) {
                if (!splitting[nd]) continue;
                const double mid = 0.5 * (lo_bound[nd] + hi_bound[nd]);
                if (left_mass[nd] < target[nd]) lo_bound[nd] = mid;
                else hi_bound[nd] = mid;
            }
        }

        // --- centre the cut in the gap it landed in ---
        // The bisection localizes the plane to the gap between the two points
        // straddling the target mass, but stops at an arbitrary spot inside
        // it. Centring -- the rule the serial build uses -- puts the plane as
        // far from either point as possible, so ownership does not hinge on
        // the last bits of a projection and a sample sitting on the boundary
        // classifies the same way.
        std::vector<double> straddle(static_cast<size_t>(n_node) * 2,
                                     std::numeric_limits<double>::lowest());
        for (int i = 0; i < n_local; ++i) {
            const int nd = node_of[i];
            if (nd < 0 || !splitting[nd]) continue;
            const double proj = projection(i, nd);
            const double cut = 0.5 * (lo_bound[nd] + hi_bound[nd]);
            if (proj < cut) {
                double& lmax = straddle[2 * static_cast<size_t>(nd)];
                lmax = std::max(lmax, proj);
            } else {
                double& neg_rmin = straddle[2 * static_cast<size_t>(nd) + 1];
                neg_rmin = std::max(neg_rmin, -proj);
            }
        }
        all_reduce(straddle.data(), static_cast<int>(straddle.size()), MPI_MAX);

        std::vector<double> cut_at(n_node, 0.0);
        for (int nd = 0; nd < n_node; ++nd) {
            if (!splitting[nd]) continue;
            const double lmax = straddle[2 * static_cast<size_t>(nd)];
            const double rmin = -straddle[2 * static_cast<size_t>(nd) + 1];
            const bool both = lmax > std::numeric_limits<double>::lowest() &&
                              rmin < std::numeric_limits<double>::max();
            cut_at[nd] = both ? 0.5 * (lmax + rmin) : 0.5 * (lo_bound[nd] + hi_bound[nd]);
        }

        // --- record the cuts and descend ---
        std::vector<Node> next;
        std::vector<int> child(static_cast<size_t>(n_node) * 2, -1);
        for (int nd = 0; nd < n_node; ++nd) {
            Node& node = level[static_cast<size_t>(nd)];
            if (!splitting[nd]) {
                node.slot->leaf = true;
                node.slot->rank = node.lo;
                continue;
            }
            const int split_rank = node.lo + (node.hi - node.lo) / 2;
            node.slot->leaf = false;
            for (int k = 0; k < 3; ++k) node.slot->axis[k] = axes[static_cast<size_t>(nd) * 3 + k];
            node.slot->threshold = cut_at[nd];
            node.slot->left.reset(new InertialCut());
            node.slot->right.reset(new InertialCut());
            child[2 * static_cast<size_t>(nd)] = static_cast<int>(next.size());
            next.push_back(Node{node.lo, split_rank, node.slot->left.get()});
            child[2 * static_cast<size_t>(nd) + 1] = static_cast<int>(next.size());
            next.push_back(Node{split_rank, node.hi, node.slot->right.get()});
        }

        for (int i = 0; i < n_local; ++i) {
            const int nd = node_of[i];
            if (nd < 0) continue;
            if (!splitting[nd]) {
                result.owner[i] = level[static_cast<size_t>(nd)].lo;
                node_of[i] = -1;
                continue;
            }
            const bool left = projection(i, nd) < cut_at[nd];
            node_of[i] = child[2 * static_cast<size_t>(nd) + (left ? 0 : 1)];
        }
        level.swap(next);
    }

    for (int i = 0; i < n_local; ++i) result.counts[static_cast<size_t>(result.owner[i])] += 1;
    if (comm != MPI_COMM_NULL) {
        MPI_Allreduce(MPI_IN_PLACE, result.counts.data(), size, MPI_INT, MPI_SUM, comm);
    }
    return result;
}

inline int ClassifyPoint(const InertialCut& tree, const double* point) {
    const InertialCut* node = &tree;
    while (!node->leaf) {
        double value = 0.0;
        for (int x = 0; x < 3; ++x) value += point[x] * node->axis[x];
        node = (value < node->threshold) ? node->left.get() : node->right.get();
    }
    return node->rank;
}

/// Marks every rank whose region comes within `radius` of `point`.
///
/// Exact, not a sample: at each cut the ball's extent along the plane normal is
/// [d - radius, d + radius] with d the signed distance to the plane, so the
/// branches it can reach are read straight off that interval and both are taken
/// when it straddles. Probing with a handful of points instead -- the box
/// corners, say -- looks conservative and is not: a region thinner than the ball
/// can pass between the probes and be missed, which is invisible on a few fat
/// partitions and starts dropping bonds as the rank count climbs.
///
/// `owners` must have one entry per rank; entries are set, never cleared, so a
/// caller can accumulate over several images before reading it.
inline void MarkRanksWithinRadius(const InertialCut& tree, const double* point, double radius,
                                  std::vector<char>& owners) {
    if (tree.leaf) {
        owners[static_cast<size_t>(tree.rank)] = 1;
        return;
    }
    double value = 0.0;
    for (int x = 0; x < 3; ++x) value += point[x] * tree.axis[x];
    const double d = value - tree.threshold;
    if (d - radius < 0.0) MarkRanksWithinRadius(*tree.left, point, radius, owners);
    if (d + radius >= 0.0) MarkRanksWithinRadius(*tree.right, point, radius, owners);
}

/// Owners of many points, walking the same planes the atoms were cut by.
inline std::vector<int> ClassifyPoints(const InertialCut& tree,
                                       const std::vector<double>& points) {
    if (points.size() % 3 != 0) {
        throw std::runtime_error("ClassifyPoints needs an (n, 3) position array.");
    }
    const int n = static_cast<int>(points.size() / 3);
    std::vector<int> owner(n, 0);
    for (int i = 0; i < n; ++i) owner[i] = ClassifyPoint(tree, &points[3 * i]);
    return owner;
}


}  // namespace atomic
}  // namespace vbcsr

#endif  // VBCSR_CORE_ATOMIC_INERTIAL_PARTITION_HPP
