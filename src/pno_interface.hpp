#pragma once
#include "functionsaver.hpp"
#include "integrals.hpp"
#include "madness_process.hpp"
#include <madness/mra/vmra.h>
#include <madness/chem/nemo.h>
#include <madness/chem/PNO.h>
#include <string>
#include <vector>
#include <utility>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace madness;

template <typename T1, typename T2> std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& v) {
    os << "(" << v.first << "," << v.second << ")";
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (auto i = 0; i < v.size(); ++i) {
        os << v[i] << " ";
    }
    os << "]";
    return os;
}

inline std::pair<size_t, char**> stringToCharPointerArray(const std::string& input) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;
    std::string current_token;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (c == '"' && (i == 0 || input[i - 1] != '\\')) { // Handle quotes
            in_quotes = !in_quotes;
            if (!in_quotes) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else if (std::isspace(c) && !in_quotes) { // Handle spaces outside quotes
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c; // Add character to the current token
        }
    }

    // Add the last token if it exists
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    // Allocate memory for the array of char* pointers
    char** charArray = new char*[tokens.size()];

    // Copy each token into the array
    for (size_t i = 0; i < tokens.size(); ++i) {
        charArray[i] = new char[tokens[i].size() + 1]; // +1 for null terminator
        std::strcpy(charArray[i], tokens[i].c_str());
    }

    return {tokens.size(), charArray};
}

// Clean up memory
inline void freeCharPointerArray(char** charArray, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        delete[] charArray[i]; // Free each string
    }
    delete[] charArray; // Free the array of pointers
}

class PNOInterface {
  public:
    PNOInterface(MadnessProcess<3>& mp, std::string argv);
    ~PNOInterface();

    SavedFct<3> get_nuclear_potential(); 
    void run(const size_t basis_size);
    void compute_scf();
    void compute_cis(const size_t n_excitations);
    void compute_cispd(const size_t basis_size);

    double get_nuclear_repulsion() const { return nuclear_repulsion; };
    std::size_t get_frozen_core_dim() const { return this->nfreeze; };
    std::vector<SavedFct<3>> get_hf_orbitals() const;
    std::vector<SavedFct<3>> get_mp2_pnos() const;
    std::vector<SavedFct<3>> get_orbitals() const;
    std::vector<std::vector<SavedFct<3>>> get_cis_x_per_root() const;
    std::vector<SavedFct<3>> get_cispd_orbitals() const;    
    std::vector<SavedFct<3>> get_sto3g() const;

  private:
    MadnessProcess<3>& madness_process;
    commandlineparser parser;

    std::shared_ptr<Nemo> nemo;
    std::vector<real_function_3d> hf_orbitals; // store HF orbitals
    real_function_3d Vnuc; // store nuclear potential

    std::vector<madness::PNOPairs> mp2_pairs;
    std::vector<real_function_3d> mp2_pnos;
    bool mp2_computed = false; // flag to indicate if MP2 PNOs were computed 

    std::vector<CC_vecfunction> cis_roots;
    std::vector<std::vector<real_function_3d>> cis_x_per_root; // cis_x_per_root to store x vectors for each root
    std::vector<std::vector<double>> cis_x_norms; // cis_x_norms to store norms for each root
    bool cis_computed = false; // flag to indicate if CIS X vectors were computed 

    std::vector<real_function_3d> cispd_pnos;
    std::vector<double> cispd_occ;
    std::vector<std::pair<size_t, size_t>> cispd_ids;
    std::vector<std::string> cispd_labels;

    std::vector<real_function_3d> basis; // HF orbitals + MP2 PNOs
    std::vector<real_function_3d> sto3g;

  protected:
    size_t nfreeze;
    double nuclear_repulsion;
    double scf_energy;
    double pno_thresh;

    std::vector<double> occ; // occ of HF + MP2 PNOs
    std::vector<std::pair<size_t, size_t>> ids; // ids of HF + MP2 PNOs
    std::vector<std::string> labels; 
};