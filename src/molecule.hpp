#pragma once

#include <madness/chem/molecule.h>

class molecule {
    public:
        molecule(){mol = madness::Molecule();}
        void add_atom(float pos_x, float pos_y, float pos_z, std::string symbol) {mol.add_atom(pos_x, pos_y, pos_z, (double)symbol_to_atomic_number(symbol), symbol_to_atomic_number(symbol));}
        std::string to_json() {return mol.to_json().dump();}
        madness::Molecule mol;
};