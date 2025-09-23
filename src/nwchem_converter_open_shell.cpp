#include "nwchem_converter_open_shell.hpp"

using namespace madness;

NWChem_Converter_open_shell::NWChem_Converter_open_shell(MadnessProcess<3>& mp) : madness_process(mp) {
    std::cout.precision(6);
}

NWChem_Converter_open_shell::~NWChem_Converter_open_shell() {
    aos.clear();
    alpha_mos.clear();
    beta_mos.clear();
}

void NWChem_Converter_open_shell::read_nwchem_file(std::string nwchem_file) {
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

    // Vnuc = create_nuclear_correlation_factor(*(madness_process.world), molecule).U2();
    // Vnuc = new Nuclear<double,3>(*(madness_process.world), molecule);
    PotentialManager pm = PotentialManager(molecule, "");
    pm.make_nuclear_potential(*(madness_process.world));
    Vnuc = pm.vnuclear();

    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();

    // Transform ao's now
    normalize(*(madness_process.world), aos);
    alpha_mos = transform(*(madness_process.world), aos, nwchem.MOs);
    beta_mos = transform(*(madness_process.world), aos, nwchem.beta_MOs);
    truncate(*(madness_process.world), alpha_mos);
    truncate(*(madness_process.world), beta_mos);
}

std::vector<SavedFct<3>> NWChem_Converter_open_shell::get_normalized_aos() {
    std::vector<SavedFct<3>> all_orbs;
    for (int i = 0; i < aos.size(); i++) {
        SavedFct<3> orb(aos[i]);
        orb.type = "ao";
        all_orbs.push_back(orb);
    }
    return all_orbs;
}

std::vector<SavedFct<3>> NWChem_Converter_open_shell::get_alpha_mos() {
    std::vector<SavedFct<3>> all_orbs;
    for (int i = 0; i < alpha_mos.size(); i++) {
        SavedFct<3> orb(alpha_mos[i]);
        orb.type = "so_alpha";
        all_orbs.push_back(orb);
    }
    return all_orbs;
}

std::vector<SavedFct<3>> NWChem_Converter_open_shell::get_beta_mos() {
    std::vector<SavedFct<3>> all_orbs;
    for (int i = 0; i < beta_mos.size(); i++) {
        SavedFct<3> orb(beta_mos[i]);
        orb.type = "so_beta";
        all_orbs.push_back(orb);
    }
    return all_orbs;
}