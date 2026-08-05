#ifndef VBCSR_SCALAR_TRAITS_HPP
#define VBCSR_SCALAR_TRAITS_HPP

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer(omp_priv = std::complex<double>(0, 0))
#endif

namespace vbcsr {

template <typename T>
struct ScalarTraits {
    using real_type = T;
    static T conjugate(const T& v) { return v; }
    // The nearest value equal to its own conjugate; identity for reals.
    static T hermitize(const T& v) { return v; }
};

template <typename T>
struct ScalarTraits<std::complex<T>> {
    using real_type = T;
    static std::complex<T> conjugate(const std::complex<T>& v) { return std::conj(v); }
    static std::complex<T> hermitize(const std::complex<T>& v) {
        return std::complex<T>(v.real(), T(0));
    }
};

} // namespace vbcsr

#endif
