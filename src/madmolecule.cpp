#include "madmolecule.hpp"

using namespace madness;

MadMolecule::MadMolecule() {
    mol = madness::Molecule();
}

void MadMolecule::add_atom(float pos_x, float pos_y, float pos_z, std::string symbol) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    pos_x *= scale;
    pos_y *= scale;
    pos_z *= scale;
    mol.add_atom(pos_x, pos_y, pos_z, (double)symbol_to_atomic_number(symbol), symbol_to_atomic_number(symbol));
}

std::string MadMolecule::to_json() {
    return mol.to_json().dump();
}


SavedFct MadMolecule::compute_nuclear_derivative(MadnessProcess& mp, const int atom, const int axis) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    madchem::MolecularDerivativeFunctor func(mol, atom, axis);
    real_function_3d op = real_factory_3d(*(mp.world)).functor(func).truncate_on_project().truncate_mode(0);
    op = scale * op; // scale the function to Angstrom
    return SavedFct(op);
}

SavedFct MadMolecule::compute_second_nuclear_derivative(MadnessProcess& mp, const int atom, const int axis1,
                                                  const int axis2) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;                                     
    madchem::MolecularSecondDerivativeFunctor func(mol, atom, axis1, axis2);
    real_function_3d op = real_factory_3d(*(mp.world)).functor(func).truncate_on_project().truncate_mode(0);
    op = scale * scale * op; // scale the function to Angstrom
    return SavedFct(op);
}

double MadMolecule::nuclear_repulsion_derivative(const int atom, const int axis) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    return scale * mol.nuclear_repulsion_derivative(atom, axis);
}

double MadMolecule::nuclear_repulsion_second_derivative(const int atom1, const int atom2, const int axis1, const int axis2) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    return scale * scale * mol.nuclear_repulsion_second_derivative(atom1, atom2, axis1, axis2);
}


SavedFct MadMolecule::get_vnuc(MadnessProcess& mp) {
    PotentialManager pm = PotentialManager(mol, "");
    pm.make_nuclear_potential(*(mp.world));
    real_function_3d Vnuc = pm.vnuclear();
    return SavedFct(Vnuc);
}