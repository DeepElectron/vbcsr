#ifndef VBCSR_CORE_RANDOM_VECTORS_HPP
#define VBCSR_CORE_RANDOM_VECTORS_HPP

// The random vectors every stochastic estimator here samples with.
//
// ONE generator, in one place. Before this there were two: a normalised
// complex Gaussian in DistVector/DistMultiVector (used by the DOS, AC, DC and
// MSD) and an inline Rademacher loop in rsatb's rs_ldos. They disagreed about
// the distribution AND about whether reproducibility was a requirement.
//
// COUNTER-BASED, and that is the whole point. The value of an entry is a hash
// of (seed, global block, row within the block, column) -- never of a thread's
// position in a loop or a rank's slice of the vector. The previous design
// seeded one std::mt19937 per OpenMP thread from std::random_device and let
// each thread fill a static slice, so the vector depended on the thread count,
// on the rank count, AND on the entropy pool: four runs of the same DOS gave
// four different answers (sum 170.8 to 202.1, peak 13.9 to 44.4), with no seed
// anywhere to pin them. Hashing a global key instead makes the draw identical
// under any decomposition, which is what "reproducible" has to mean for a
// distributed code -- same answer on 1 rank and on 64, at any thread count.
//
// A block never splits across ranks, so (global block, row within block) is a
// global name for an entry that costs nothing to compute -- no prefix sum over
// block sizes, no O(N) table on every rank.
//
// On the distribution: Rademacher and random phase have |v|^2 = N EXACTLY, so
// normalising them is a scale by 1/sqrt(N) with no communication, where the
// Gaussian needs a dot product and its global reduction. They also carry less
// variance. For Hutchinson trace estimation,
//
//     Var(Rademacher) = 2(||A||_F^2 - sum_i |A_ii|^2)
//     Var(Gaussian)   = 2||A||_F^2
//     Var(sphere)     = 2(N||A||_F^2 - (tr A)^2)/(N+2)
//
// and sum_i |A_ii|^2 >= (tr A)^2 / N by Cauchy-Schwarz, with equality only for
// a flat diagonal. For a spectral density the diagonal IS the site-projected
// DOS, so the three coincide only when every site is equivalent and Rademacher
// wins whenever they are not -- surfaces, defects, alloys. Random phase is its
// complex analogue and is proven optimal within a wide class for this problem
// (Iitaka & Ebisuzaki, cond-mat/0401202).

#include "detail/storage/numa_buffer.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace vbcsr {

enum class RandomKind {
    Rademacher,   ///< +-1. Real draw; the default for real scalars.
    RandomPhase,  ///< exp(i theta). Complex scalars only.
    Gaussian,     ///< N(0,1) per real component. Kept for cross-checks: it is
                  ///< the textbook estimator and the only one of the three
                  ///< whose norm is not deterministic.
};

namespace detail {

/// splitmix64. A counter hash, not a stream: the n-th value is computed from n
/// directly, which is what lets any rank produce any entry without ordering.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/// Hash of the full key. Mixed one field at a time so that neighbouring blocks,
/// neighbouring rows and neighbouring columns all decorrelate.
inline uint64_t entry_hash(uint64_t seed, uint64_t block, uint64_t row,
                           uint64_t col) {
    uint64_t h = splitmix64(seed + 0x9e3779b97f4a7c15ULL);
    h = splitmix64(h ^ (block * 0xbf58476d1ce4e5b9ULL));
    h = splitmix64(h ^ (row * 0x94d049bb133111ebULL));
    return splitmix64(h ^ (col * 0xd6e8feb86659fd93ULL));
}

/// Uniform in [0,1), from the top 53 bits.
inline double uniform01(uint64_t h) {
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
}

/// Box-Muller from two hashes. Only Gaussian needs it.
inline double normal_from(uint64_t h1, uint64_t h2) {
    const double u1 = std::max(uniform01(h1), 1e-300);
    const double u2 = uniform01(h2);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

}  // namespace detail

/// One entry of a random vector, addressed by its GLOBAL name.
///
/// Deterministic in (kind, seed, block, row, col) alone: no rank, no thread, no
/// call order. Two runs on different rank counts produce the same vector.
template <typename T>
inline T random_entry(RandomKind kind, uint64_t seed, uint64_t block,
                      uint64_t row, uint64_t col) {
    const uint64_t h = detail::entry_hash(seed, block, row, col);
    if constexpr (std::is_same_v<T, std::complex<double>> ||
                  std::is_same_v<T, std::complex<float>>) {
        using R = typename T::value_type;
        switch (kind) {
            case RandomKind::RandomPhase: {
                const double theta = 2.0 * M_PI * detail::uniform01(h);
                return T(static_cast<R>(std::cos(theta)),
                         static_cast<R>(std::sin(theta)));
            }
            case RandomKind::Rademacher:
                // Real +-1 held in a complex scalar: the imaginary lane stays
                // zero, so it costs the same matvecs as RandomPhase and buys
                // strictly less. Offered only so a caller can reproduce a real
                // run inside a complex pipeline.
                return T(static_cast<R>((h >> 63) ? 1.0 : -1.0), R(0));
            case RandomKind::Gaussian: {
                const uint64_t h2 = detail::splitmix64(h);
                // 1/sqrt(2) per lane so that E|v_i|^2 = 1, matching the other
                // two kinds -- otherwise the kinds would not be interchangeable.
                const double s = 1.0 / std::sqrt(2.0);
                return T(static_cast<R>(s * detail::normal_from(h, h2)),
                         static_cast<R>(s * detail::normal_from(h2,
                                                 detail::splitmix64(h2))));
            }
        }
        return T(0);
    } else {
        switch (kind) {
            case RandomKind::Rademacher:
                return static_cast<T>((h >> 63) ? 1.0 : -1.0);
            case RandomKind::RandomPhase:
                throw std::runtime_error(
                    "RandomKind::RandomPhase needs a complex scalar type; use "
                    "Rademacher for a real one.");
            case RandomKind::Gaussian:
                return static_cast<T>(
                    detail::normal_from(h, detail::splitmix64(h)));
        }
        return T(0);
    }
}

/// Does this kind have |v|^2 = N exactly, so that normalising needs no
/// communication? True for every kind whose entries have unit modulus.
inline bool has_deterministic_norm(RandomKind kind) {
    return kind != RandomKind::Gaussian;
}

/// The kind to use when the caller has no opinion: the lowest-variance draw
/// the scalar type supports.
template <typename T>
inline RandomKind default_random_kind() {
    if constexpr (std::is_same_v<T, std::complex<double>> ||
                  std::is_same_v<T, std::complex<float>>) {
        return RandomKind::RandomPhase;
    } else {
        return RandomKind::Rademacher;
    }
}

}  // namespace vbcsr

#endif  // VBCSR_CORE_RANDOM_VECTORS_HPP
