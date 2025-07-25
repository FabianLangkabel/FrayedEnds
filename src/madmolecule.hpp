#pragma once

#include "functionsaver.hpp"
#include "MadnessProcess.hpp"
#include <madness/mra/mra.h>
#include <madness/chem/molecule.h>
#include <madness/chem/molecular_functors.h>
#include <madness/chem/potentialmanager.h>
#include <cstdio>
#include <unistd.h>

class MadMolecule {
  public:
    MadMolecule();

    void add_atom(float pos_x, float pos_y, float pos_z, std::string symbol);

    std::string to_json();

    SavedFct compute_nuclear_derivative(MadnessProcess& mp, const int atom, const int axis);

    SavedFct get_vnuc(MadnessProcess& mp);

    madness::Molecule mol;

  private:
    real_function_3d Vnuc;
};