#include "nwchem_converter.hpp"

using namespace madness;

NWChem_Converter::NWChem_Converter(MadnessProcess& mp) : madness_process(mp) {
    std::cout.precision(6);
}

NWChem_Converter::~NWChem_Converter() {
    aos.clear();
    mos.clear();
}

void NWChem_Converter::read_nwchem_file(std::string nwchem_file) {
    std::ostream bad(nullptr);
    // slymer::NWChem_Interface nwchem(nwchem_file, bad);
    slymer::NWChem_Interface nwchem(nwchem_file, std::cout);

    nwchem.read(slymer::Properties::Atoms | slymer::Properties::Basis | slymer::Properties::Energies |
                slymer::Properties::MOs | slymer::Properties::Occupancies);

    // Cast the 'basis_set' into a gaussian basis
    // and iterate over it
    int i = 0;
    for (auto basis : slymer::cast_basis<slymer::GaussianFunction>(nwchem.basis_set)) {
        // Get the center of gaussian as its special point
        std::vector<coord_3d> centers;
        coord_3d r;
        r[0] = basis.get().center[0];
        r[1] = basis.get().center[1];
        r[2] = basis.get().center[2];
        centers.push_back(r);

        // Now make the function
        aos.push_back(
            factoryT(*(madness_process.world)).functor(functorT(new slymer::Gaussian_Functor(basis.get(), centers))));
    }

    auto molecule = madness::Molecule();
    for (auto atom : nwchem.atoms) {
        molecule.add_atom(atom.position[0], atom.position[1], atom.position[2],
                          (double)symbol_to_atomic_number(atom.symbol), symbol_to_atomic_number(atom.symbol));
    }

    // Vnuc = new Nuclear<double,3>(*(madness_process.world), molecule);
    // nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();

    // Transform ao's now
    normalize(*(madness_process.world), aos);
    mos = transform(*(madness_process.world), aos, nwchem.MOs);
    truncate(*(madness_process.world), mos);
}

std::vector<SavedFct> NWChem_Converter::GetNormalizedAOs() {
    std::vector<SavedFct> all_orbs;
    for (int i = 0; i < aos.size(); i++) {
        SavedFct orb(aos[i]);
        orb.type = "ao";
        all_orbs.push_back(orb);
    }
    return all_orbs;
}

std::vector<SavedFct> NWChem_Converter::GetMOs() {
    std::vector<SavedFct> all_orbs;
    for (int i = 0; i < mos.size(); i++) {
        SavedFct orb(mos[i]);
        orb.type = "mo";
        all_orbs.push_back(orb);
    }
    return all_orbs;
}
