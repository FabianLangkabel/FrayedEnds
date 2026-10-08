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
// The calculation runs on the FunctionDefaults of the given MadnessProcess, which are
// restored after the calculation.
class HartreeFock {
  public:
    // parameters_json: MADNESS "dft" calculation parameters as JSON object, e.g. {"econv": 1e-6, "protocol": [1e-6]}
    HartreeFock(MadnessProcess<3>& mp, const MolecularGeometry& molecule, const std::string& parameters_json);

    // guess for the occupied orbitals, beta orbitals are only used in spin-unrestricted calculations
    void set_initial_orbitals(const std::vector<SavedFct<3>>& alpha_orbitals, const std::vector<double>& alpha_energies,
                              const std::vector<SavedFct<3>>& beta_orbitals, const std::vector<double>& beta_energies);

    double solve();

    std::vector<SavedFct<3>> get_alpha_orbitals() const;
    std::vector<SavedFct<3>> get_beta_orbitals() const;
    std::vector<double> get_alpha_orbital_energies() const;
    std::vector<double> get_beta_orbital_energies() const;
    // minimal (sto-3g) AO basis projected into MRA, used by the SCF for orbital analysis
    std::vector<SavedFct<3>> get_minimal_basis() const;
    double get_energy() const;
    SavedFct<3> get_vnuc() const;
    double get_nuclear_repulsion() const;
    bool is_spin_restricted() const { return param.spin_restricted(); }

  private:
    MadnessProcess<3>& madness_process;
    Molecule molecule;
    CalculationParameters param;
    std::shared_ptr<SCF> calc;

    std::vector<real_function_3d> guess_alpha;
    std::vector<real_function_3d> guess_beta;
    std::vector<double> guess_alpha_energies;
    std::vector<double> guess_beta_energies;

    void install_initial_orbitals(World& world);
    void check_solved() const;
};
