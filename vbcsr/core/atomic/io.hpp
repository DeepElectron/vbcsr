#ifndef VBCSR_CORE_ATOMIC_IO_HPP
#define VBCSR_CORE_ATOMIC_IO_HPP

// Atomic structure input: positions, species, cell, periodicity.
//
// A Structure carries no orbital, cutoff, or model information: it is geometry
// and nothing else. What the atoms mean is the caller's business, which is what
// lets AtomicData::from_file read one without being told anything about the
// basis beyond the per-type tables it is handed.

#include "periodic_table.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbcsr {
namespace atomic {

inline std::vector<std::string> SplitWhitespace(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(line);
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

/// Atomic geometry in Angstrom.
///
/// Positions may be given anywhere -- inside the cell, outside it, or in a
/// mixture. POSCAR, XYZ and CIF all permit out-of-cell coordinates and none of
/// them is required to wrap.
///
/// The convention, held end to end: coordinates and atom order are kept exactly
/// as supplied, all the way into the graph. Nothing is wrapped, sorted, or
/// renumbered behind the caller's back, because a Hamiltonian file's atom
/// indexing and R-vector convention refer to the coordinates its author wrote
/// down -- moving an atom would silently point the file lookup at a different
/// bond. Edge shift vectors are therefore expressed against these same
/// coordinates, so a bond is r_j + R - r_i whatever the input looked like.
/// (Neighbour searching still wraps internally to bin atoms; it undoes the
/// translation before reporting a shift.)
struct Structure {
    std::vector<double> pos;   // 3 * n_atom, cartesian
    std::vector<int> z;        // n_atom, atomic numbers
    std::vector<double> cell;  // 9, row-major lattice vectors
    std::vector<bool> pbc;     // 3
    int n_atom = 0;

    /// Returns the distinct species in the order AtomicData assigns type indices.
    std::vector<std::string> TypeSymbols() const {
        return UniqueSymbolsFromAtomicNumbers(z);
    }
};

inline Structure ReadPoscar(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

    Structure data;
    std::string line;
    std::vector<std::string> tokens;

    // 1. Comment
    if (!std::getline(file, line)) throw std::runtime_error("Empty POSCAR");

    // 2. Scale
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (scale)");
    const double scale = std::stod(line);

    // 3-5. Cell
    data.cell.resize(9);
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (cell)");
        tokens = SplitWhitespace(line);
        if (tokens.size() < 3) throw std::runtime_error("Invalid cell line");
        for (int j = 0; j < 3; ++j) data.cell[3 * i + j] = std::stod(tokens[j]) * scale;
    }
    data.pbc = {true, true, true};

    // 6. Elements (VASP 5) or counts (VASP 4)
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (species/counts)");
    tokens = SplitWhitespace(line);
    if (tokens.empty()) throw std::runtime_error("Empty species/counts line");

    std::vector<std::string> elements;
    std::vector<int> counts;

    bool vasp5 = false;
    if (std::isalpha(static_cast<unsigned char>(tokens[0][0]))) {
        vasp5 = true;
        elements = tokens;
        if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (counts)");
        tokens = SplitWhitespace(line);
    }

    for (const auto& token : tokens) counts.push_back(std::stoi(token));

    data.n_atom = 0;
    for (int count : counts) data.n_atom += count;

    data.pos.resize(static_cast<size_t>(data.n_atom) * 3);
    data.z.resize(data.n_atom);

    // 7. Coordinate type
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (coord type)");
    const bool direct = (line[0] == 'D' || line[0] == 'd');

    // 8. Positions
    int idx = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        int z_val = 0;
        if (vasp5 && i < elements.size()) {
            z_val = AtomicNumber(elements[i]);
        } else {
            z_val = static_cast<int>(i) + 1;  // dummy Z when the file has no species line
        }

        for (int j = 0; j < counts[i]; ++j) {
            if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (positions)");
            tokens = SplitWhitespace(line);
            if (tokens.size() < 3) throw std::runtime_error("Invalid position line");
            const double x = std::stod(tokens[0]);
            const double y = std::stod(tokens[1]);
            const double z_coord = std::stod(tokens[2]);

            if (direct) {
                data.pos[3 * idx] = x * data.cell[0] + y * data.cell[3] + z_coord * data.cell[6];
                data.pos[3 * idx + 1] = x * data.cell[1] + y * data.cell[4] + z_coord * data.cell[7];
                data.pos[3 * idx + 2] = x * data.cell[2] + y * data.cell[5] + z_coord * data.cell[8];
            } else {
                data.pos[3 * idx] = x * scale;
                data.pos[3 * idx + 1] = y * scale;
                data.pos[3 * idx + 2] = z_coord * scale;
            }
            data.z[idx] = z_val;
            ++idx;
        }
    }

    return data;
}

inline Structure ReadXyz(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

    Structure data;
    std::string line;

    // 1. Atom count
    if (!std::getline(file, line)) throw std::runtime_error("Empty XYZ");
    try {
        data.n_atom = std::stoi(line);
    } catch (...) {
        throw std::runtime_error("Invalid XYZ header (N atoms)");
    }

    // 2. Comment line, optionally carrying an extended-XYZ Lattice="..." record
    if (!std::getline(file, line)) throw std::runtime_error("Truncated XYZ (comment)");

    data.cell = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    data.pbc = {false, false, false};

    const size_t lattice_pos = line.find("Lattice=\"");
    if (lattice_pos != std::string::npos) {
        std::string lattice = line.substr(lattice_pos + 9);
        const size_t end_pos = lattice.find('"');
        if (end_pos != std::string::npos) {
            lattice = lattice.substr(0, end_pos);
            const std::vector<std::string> tokens = SplitWhitespace(lattice);
            if (tokens.size() >= 9) {
                for (int i = 0; i < 9; ++i) data.cell[i] = std::stod(tokens[i]);
                data.pbc = {true, true, true};
            }
        }
    }

    data.pos.resize(static_cast<size_t>(data.n_atom) * 3);
    data.z.resize(data.n_atom);

    for (int i = 0; i < data.n_atom; ++i) {
        if (!std::getline(file, line)) throw std::runtime_error("Truncated XYZ (positions)");
        const std::vector<std::string> tokens = SplitWhitespace(line);
        if (tokens.size() < 4) continue;

        data.z[i] = AtomicNumber(tokens[0]);
        data.pos[3 * i] = std::stod(tokens[1]);
        data.pos[3 * i + 1] = std::stod(tokens[2]);
        data.pos[3 * i + 2] = std::stod(tokens[3]);
    }

    return data;
}

/// Reads a structure, inferring the format from the filename when unspecified.
///
/// Args:
///   filename: Path to a POSCAR/.vasp or .xyz file.
///   format: "POSCAR" or "XYZ"; empty to infer from `filename`.
inline Structure ReadStructure(const std::string& filename, const std::string& format = "") {
    std::string resolved = format;
    if (resolved.empty()) {
        if (filename.find(".vasp") != std::string::npos ||
            filename.find("POSCAR") != std::string::npos) {
            resolved = "POSCAR";
        } else if (filename.find(".xyz") != std::string::npos) {
            resolved = "XYZ";
        } else {
            throw std::runtime_error("Unknown structure format for file: " + filename);
        }
    }

    if (resolved == "POSCAR") return ReadPoscar(filename);
    if (resolved == "XYZ") return ReadXyz(filename);
    throw std::runtime_error("Unsupported structure format: " + resolved);
}

}  // namespace atomic
}  // namespace vbcsr

#endif  // VBCSR_CORE_ATOMIC_IO_HPP
