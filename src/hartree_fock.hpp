#pragma once

#include "functionsaver.hpp"
#include "madness_process.hpp"
#include "moleculargeometry.hpp"
#include <madness/chem/SCF.h>
#include <memory>
#include <string>
#include <vector>

using namespace madness;

// Hartree-Fock calculation with the MADNESS SCF solver (the solver behind moldft).
// The occupied orbitals can be started from user-provided guess orbitals (e.g. NWChem MOs
// translated into MRA with the NWChem_Converter) or from the default MADNESS AO guess.
class HartreeFock {
  public:
    // parameters_json: MADNESS "dft" calculation parameters as JSON object, e.g. {"econv": 1e-6, "protocol": [1e-6]}
    // localize_result: "localize" method applied once to the converged orbitals (empty: no final rotation)
    HartreeFock(MadnessProcess<3>& mp, const MolecularGeometry& molecule, const std::string& parameters_json,
                const std::string& localize_result = "");

    // guess for the occupied orbitals, beta orbitals are only used in spin-unrestricted calculations
    void set_initial_orbitals(const std::vector<SavedFct<3>>& alpha_orbitals, const std::vector<double>& alpha_energies,
                              const std::vector<SavedFct<3>>& beta_orbitals, const std::vector<double>& beta_energies);

    double solve();

    std::vector<SavedFct<3>> get_alpha_orbitals() const { return to_saved(scf().amo, "hf_alpha"); }
    std::vector<SavedFct<3>> get_beta_orbitals() const;
    std::vector<double> get_alpha_orbital_energies() const { return to_vector(scf().aeps); }
    std::vector<double> get_beta_orbital_energies() const;
    // minimal (sto-3g) AO basis projected into MRA, used by the SCF for orbital analysis
    std::vector<SavedFct<3>> get_minimal_basis() const { return to_saved(scf().ao, "sto-3g"); }
    double get_energy() const { return scf().current_energy; }
    SavedFct<3> get_vnuc() const { return SavedFct<3>(scf().potentialmanager->vnuclear()); }
    double get_nuclear_repulsion() const { return molecule.nuclear_repulsion_energy(); }
    bool is_spin_restricted() const { return param.spin_restricted(); }

  private:
    MadnessProcess<3>& madness_process;
    Molecule molecule;
    CalculationParameters param;
    std::shared_ptr<SCF> calc;
    std::string localize_result; // empty: keep the orbitals of the last SCF iteration

    std::vector<real_function_3d> guess_alpha, guess_beta;
    std::vector<double> guess_alpha_energies, guess_beta_energies;

    void install_initial_orbitals(World& world);
    void rotate_final_orbitals(World& world);
    const SCF& scf() const; // throws if solve() was not called yet
    // the first n entries (default: all)
    static std::vector<SavedFct<3>> to_saved(const vecfuncT& f, const std::string& info, std::size_t n = -1);
    static std::vector<double> to_vector(const Tensor<double>& t, std::size_t n = -1);
};
