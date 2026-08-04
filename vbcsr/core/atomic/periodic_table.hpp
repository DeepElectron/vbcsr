#ifndef VBCSR_CORE_ATOMIC_PERIODIC_TABLE_HPP
#define VBCSR_CORE_ATOMIC_PERIODIC_TABLE_HPP

// Element symbol <-> atomic number mapping.
//
// It sits beside AtomicData rather than in a chemistry package because the
// structure readers here need it and because AtomicData's own type ordering is
// defined in terms of it: types are the distinct atomic numbers in ascending
// order, so whoever supplies per-type cutoffs and orbital counts must agree
// with UniqueSymbolsFromAtomicNumbers below. One table, both directions, so
// they cannot drift apart.

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbcsr {
namespace atomic {

inline std::string TrimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline std::string LowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Element symbols indexed by atomic number; index 0 is unused.
inline const std::vector<std::string>& ElementSymbols() {
    static const std::vector<std::string> symbols = {
        "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
        "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
        "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
        "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
        "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
        "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    return symbols;
}

/// Returns the atomic number for a symbol, or 0 when unrecognized.
///
/// A decimal string is accepted and returned as-is, so both "C" and "6"
/// resolve to carbon. Structure readers rely on the 0 return to signal an
/// unlabeled species; callers that require a real element should use
/// `CanonicalSymbol`, which rejects it.
inline int AtomicNumber(const std::string& raw_symbol) {
    static const std::map<std::string, int> table = [] {
        std::map<std::string, int> result;
        const auto& symbols = ElementSymbols();
        for (size_t z = 1; z < symbols.size(); ++z) {
            result.emplace(symbols[z], static_cast<int>(z));
        }
        return result;
    }();

    const std::string symbol = TrimCopy(raw_symbol);
    auto it = table.find(symbol);
    if (it != table.end()) {
        return it->second;
    }
    try {
        return std::stoi(symbol);
    } catch (...) {
        return 0;
    }
}

/// Normalizes any accepted species token to a canonical element symbol.
///
/// Args:
///   raw_symbol: Element symbol or decimal atomic number, e.g. "C" or "6".
/// Returns:
///   The canonical symbol, e.g. "C".
/// Throws:
///   std::runtime_error if the token names no supported element.
inline std::string CanonicalSymbol(const std::string& raw_symbol) {
    const std::string trimmed = TrimCopy(raw_symbol);
    if (trimmed.empty()) {
        throw std::runtime_error("Species symbol must not be empty.");
    }

    const int z = AtomicNumber(trimmed);
    const auto& symbols = ElementSymbols();
    if (z <= 0 || z >= static_cast<int>(symbols.size()) || symbols[z].empty()) {
        throw std::runtime_error("Unsupported species symbol: " + raw_symbol);
    }
    return symbols[z];
}

/// Returns the canonical symbols of the distinct species, in ascending Z.
///
/// The order matters: `AtomicData` assigns atom type indices by ascending
/// atomic number, so this is the ordering every `BasisSpec` must agree with.
inline std::vector<std::string> UniqueSymbolsFromAtomicNumbers(const std::vector<int>& z) {
    std::vector<int> unique_z = z;
    std::sort(unique_z.begin(), unique_z.end());
    unique_z.erase(std::unique(unique_z.begin(), unique_z.end()), unique_z.end());

    std::vector<std::string> symbols;
    symbols.reserve(unique_z.size());
    for (int value : unique_z) {
        symbols.push_back(CanonicalSymbol(std::to_string(value)));
    }
    return symbols;
}

}  // namespace atomic
}  // namespace vbcsr

#endif  // VBCSR_CORE_ATOMIC_PERIODIC_TABLE_HPP
