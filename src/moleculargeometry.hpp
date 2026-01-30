#pragma once

#include "functionsaver.hpp"
#include "madness_process.hpp"
#include <madness/mra/mra.h>
#include <madness/chem/molecule.h>
#include <madness/chem/molecular_functors.h>
#include <madness/chem/potentialmanager.h>
#include <cstdio>
#include <unistd.h>

class MolecularGeometry {
  public:
    MolecularGeometry(std::string units);
    std::string units;

    void add_atom(double pos_x, double pos_y, double pos_z, std::string symbol);

    std::string to_json();

    SavedFct<3> compute_nuclear_derivative(MadnessProcess<3>& mp, const int atom, const int axis);
    SavedFct<3> compute_second_nuclear_derivative(MadnessProcess<3>& mp, const int atom, const int axis1,
                                                  const int axis2);
    double nuclear_repulsion_derivative(const int atom, const int axis);
    double nuclear_repulsion_second_derivative(const int atom1, const int atom2, const int axis1, const int axis2);

    SavedFct<3> get_vnuc(MadnessProcess<3>& mp);

    double get_nuclear_repulsion() const { return mol.nuclear_repulsion_energy(); }
    double get_nuclear_charge() const { return mol.total_nuclear_charge(); }

    int get_core_n_electrons() const {
        int result = 0;
        for (auto atom : mol.get_atoms()) {
            auto n = atom.get_atomic_number();
            std::cout << "atom " << atom.get_atomic_number() << "\n";
            if (n > 2)
                result = 2;
            if (n > 10)
                result = 10;
            if (n > 18)
                result = 18;
            if (n > 36)
                result = 36;
            if (n > 54)
                result = 54;
        }
        return result;
    }

    madness::Molecule mol;

  private:
    real_function_3d Vnuc;
};
