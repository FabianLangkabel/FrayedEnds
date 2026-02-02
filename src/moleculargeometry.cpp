#include "moleculargeometry.hpp"

using namespace madness;

MolecularGeometry::MolecularGeometry(std::string units) : units(units) {
    if (units != "angstrom" && units != "bohr") {
        std::cout << "Warning: units of molecular geometry not recognized, assuming angstrom." << std::endl;
        this->units = "angstrom";
    }
    mol = madness::Molecule();
}

void MolecularGeometry::add_atom(double pos_x, double pos_y, double pos_z, std::string symbol) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    if (units == "angstrom") {
        pos_x *= scale;
        pos_y *= scale;
        pos_z *= scale;
    }
    mol.add_atom(pos_x, pos_y, pos_z, (double)symbol_to_atomic_number(symbol), symbol_to_atomic_number(symbol));
}

std::string MolecularGeometry::to_json() {
    return mol.to_json().dump();
}

SavedFct<3> MolecularGeometry::molecular_potential_derivative(MadnessProcess<3>& mp, const int atom, const int axis) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    madchem::MolecularDerivativeFunctor func(mol, atom, axis);
    real_function_3d op = real_factory_3d(*(mp.world)).functor(func).truncate_on_project().truncate_mode(0);
    if (units == "angstrom") {
        op = scale * op; // scale the function to Angstrom (units of the resulting function are hartree/angstrom)
    }
    return SavedFct<3>(op);
}

SavedFct<3> MolecularGeometry::molecular_potential_second_derivative(MadnessProcess<3>& mp, const int atom, const int axis1,
                                                                 const int axis2) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    madchem::MolecularSecondDerivativeFunctor func(mol, atom, axis1, axis2);
    real_function_3d op = real_factory_3d(*(mp.world)).functor(func).truncate_on_project().truncate_mode(0);
    if (units == "angstrom") {
        op = scale * scale *
             op; // scale the function to Angstrom (units of the resulting function are hartree/angstrom**2)
    }
    return SavedFct<3>(op);
}

double MolecularGeometry::nuclear_repulsion_derivative(const int atom, const int axis) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    double result = mol.nuclear_repulsion_derivative(atom, axis);
    if (units == "angstrom") {
        result = scale * result; // scale to Angstrom (units of the resulting function are hartree/angstrom)
    }
    return result;
}

double MolecularGeometry::nuclear_repulsion_second_derivative(const int atom1, const int atom2, const int axis1,
                                                              const int axis2) {
    double scale = 1e-10 / madness::constants::atomic_unit_of_length;
    double result = mol.nuclear_repulsion_second_derivative(atom1, atom2, axis1, axis2);
    if (units == "angstrom") {
        result = scale * scale * result; // scale to Angstrom (units of the resulting function are hartree/angstrom**2)
    }
    return result;
}

SavedFct<3> MolecularGeometry::get_vnuc(MadnessProcess<3>& mp) {
    PotentialManager pm = PotentialManager(mol, "");
    pm.make_nuclear_potential(*(mp.world));
    real_function_3d Vnuc = pm.vnuclear();
    return SavedFct<3>(Vnuc);
}