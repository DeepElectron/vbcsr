// The vendor's grouped batch GEMM must accumulate exactly what the intrinsic
// row-major kernel accumulates.
//
// Nothing else covers it. The batch is selected by block size and scalar type
// (common.hpp's vendor_batch_profitable), and every other test here runs block
// sizes of 1-3, so the vendor call is never reached from a matrix-level test.
// It is reached in production -- both stages of the fused triple product and
// the square polynomial flush their contraction groups through it -- so the two
// paths disagreeing would be a silent wrong answer at large block sizes only.
//
// This drives the kernel directly rather than through spmm, because routing to
// it depends on rank count, page count and block size all at once; a
// matrix-level test that looked like it covered this would mostly not.

// Test assertions must stay active in Release builds.
#undef NDEBUG

#include "../block_csr.hpp"
#include "../detail/ops/spmm/common.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace vbcsr;

namespace {

template <typename T>
T draw(std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-0.75, 0.75);
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return T(dist(gen), dist(gen));
    }
    return static_cast<T>(dist(gen));
}

template <typename T>
const char* dtype_name() {
    if constexpr (std::is_same_v<T, double>) return "double";
    return "complex<double>";
}

/// One contraction group: a single left operand against `n` right operands with
/// distinct destinations, which is the only shape the batch is ever handed.
template <typename T>
void check_group_matches_loop(int bs, size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    const size_t block_elems = static_cast<size_t>(bs) * static_cast<size_t>(bs);

    std::vector<T> a(block_elems);
    for (T& value : a) value = draw<T>(gen);

    std::vector<std::vector<T>> b(n, std::vector<T>(block_elems));
    for (auto& block : b) {
        for (T& value : block) value = draw<T>(gen);
    }

    // Seeded non-zero, because the batch runs with beta = 1: it must ADD to the
    // accumulator, not overwrite it. A zeroed destination would pass either way.
    std::vector<std::vector<T>> c(n, std::vector<T>(block_elems));
    for (auto& block : c) {
        for (T& value : block) value = draw<T>(gen);
    }
    std::vector<std::vector<T>> expected = c;

    std::vector<const T*> b_ptrs;
    std::vector<T*> c_ptrs;
    for (size_t k = 0; k < n; ++k) {
        b_ptrs.push_back(b[k].data());
        c_ptrs.push_back(c[k].data());
    }

    // Must be true here: the override is forced on, T is one of the two types
    // the batch supports, and n is at or above kMinBatch. If this ever fails
    // the comparison below would be vacuous -- it would be checking the
    // intrinsic kernel against itself, which is how the earlier attempt at
    // covering this path fooled itself.
    const bool took_vendor =
        detail::grouped_block_gemm_batch<T>(bs, a.data(), b_ptrs, c_ptrs);
    assert(took_vendor);

    for (size_t k = 0; k < n; ++k) {
        detail::fused_gemm_accumulate<T>(expected[k].data(), a.data(),
                                         b[k].data(), bs, bs, bs);
    }

    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < block_elems; ++i) {
            const double diff = std::abs(c[k][i] - expected[k][i]);
            const double tol = 1e-11 + 1e-10 * std::max(1.0, std::abs(expected[k][i]));
            if (diff > tol) {
                std::cerr << "vendor batch disagrees with the intrinsic kernel:"
                          << " type=" << dtype_name<T>() << " bs=" << bs
                          << " block=" << k << " element=" << i
                          << " got=" << c[k][i] << " expected=" << expected[k][i]
                          << std::endl;
            }
            assert(diff <= tol);
        }
    }
    std::cout << "  vendor batch matches for " << dtype_name<T>()
              << " bs=" << bs << " n=" << n << std::endl;
}

/// Below kMinBatch the call setup outweighs the gain, so the batch declines and
/// leaves the destinations alone for the caller to loop over. Pinned because
/// "declines" and "declines without having written anything" are different.
template <typename T>
void check_small_group_declines(int bs) {
    const size_t block_elems = static_cast<size_t>(bs) * static_cast<size_t>(bs);
    std::mt19937 gen(7);

    std::vector<T> a(block_elems);
    for (T& value : a) value = draw<T>(gen);
    std::vector<std::vector<T>> b(3, std::vector<T>(block_elems, T(1.0)));
    std::vector<std::vector<T>> c(3, std::vector<T>(block_elems, T(0.25)));
    const std::vector<std::vector<T>> before = c;

    std::vector<const T*> b_ptrs;
    std::vector<T*> c_ptrs;
    for (size_t k = 0; k < 3; ++k) {
        b_ptrs.push_back(b[k].data());
        c_ptrs.push_back(c[k].data());
    }

    assert(!detail::grouped_block_gemm_batch<T>(bs, a.data(), b_ptrs, c_ptrs));
    assert(c == before);
}

} // namespace

int main(int argc, char** argv) {
    // Before the first gate call: vendor_batch_profitable resolves the override
    // once per process, so this has to be in place ahead of any of it. Forcing
    // it on is the point -- the gate's DEFAULT is a performance policy, while
    // what is under test is that the kernel it selects is correct wherever it
    // does get selected.
    setenv("VBCSR_SPGEMM_BATCH", "1", 1);

    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    if (rank == 0) {
        std::cout << "grouped batch vs intrinsic kernel:" << std::endl;
    }
    for (const int bs : {8, 13, 21, 32}) {
        check_group_matches_loop<double>(bs, 6, 11u + static_cast<unsigned>(bs));
        check_group_matches_loop<std::complex<double>>(bs, 6,
                                                       97u + static_cast<unsigned>(bs));
    }
    check_group_matches_loop<double>(21, 4, 5u);
    check_group_matches_loop<std::complex<double>>(21, 4, 6u);

    check_small_group_declines<double>(21);
    check_small_group_declines<std::complex<double>>(21);
#else
    if (rank == 0) {
        std::cout << "built without VBCSR_BLAS_HAS_BATCH_GEMM: no vendor batch"
                  << std::endl;
    }
#endif

    if (rank == 0) {
        std::cout << "test_vendor_batch_gemm PASSED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
