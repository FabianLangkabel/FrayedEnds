#include "madMolecule.hpp"

using namespace madness;

MadMolecule::MadMolecule() {
    mol = madness::Molecule();
}

void MadMolecule::add_atom(float pos_x, float pos_y, float pos_z, std::string symbol) {
    mol.add_atom(pos_x, pos_y, pos_z, (double)symbol_to_atomic_number(symbol), symbol_to_atomic_number(symbol));
}

std::string MadMolecule::to_json() {
    return mol.to_json().dump();
}

SavedFct MadMolecule::compute_nuclear_derivative(MadnessProcess& mp, const int atom, const int axis) {
    madchem::MolecularDerivativeFunctor func(mol, atom, axis);
    real_function_3d op = real_factory_3d(*(mp.world)).functor(func).nofence().truncate_on_project().truncate_mode(0);
    return SavedFct(op);
}

SavedFct MadMolecule::get_vnuc(MadnessProcess& mp) {
    PotentialManager pm = PotentialManager(mol, "");
    pm.make_nuclear_potential(*(mp.world));
    real_function_3d Vnuc = pm.vnuclear();
    return SavedFct(Vnuc);
}