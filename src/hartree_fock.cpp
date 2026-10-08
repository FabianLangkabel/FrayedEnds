#include "hartree_fock.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace madness;

namespace {

// SCF::set_protocol overwrites the global FunctionDefaults<3>, this guard restores the
// state of the MadnessProcess after the SCF calculation (also if the calculation throws)
class FunctionDefaultsGuard {
  public:
    FunctionDefaultsGuard()
        : k(FunctionDefaults<3>::get_k()), thresh(FunctionDefaults<3>::get_thresh()),
          initial_level(FunctionDefaults<3>::get_initial_level()),
          truncate_mode(FunctionDefaults<3>::get_truncate_mode()), refine(FunctionDefaults<3>::get_refine()),
          autorefine(FunctionDefaults<3>::get_autorefine()),
          truncate_on_project(FunctionDefaults<3>::get_truncate_on_project()),
          apply_randomize(FunctionDefaults<3>::get_apply_randomize()),
          project_randomize(FunctionDefaults<3>::get_project_randomize()), cell(copy(FunctionDefaults<3>::get_cell())) {
    }

    ~FunctionDefaultsGuard() {
        FunctionDefaults<3>::set_k(k);
        FunctionDefaults<3>::set_thresh(thresh);
        FunctionDefaults<3>::set_initial_level(initial_level);
        FunctionDefaults<3>::set_truncate_mode(truncate_mode);
        FunctionDefaults<3>::set_refine(refine);
        FunctionDefaults<3>::set_autorefine(autorefine);
        FunctionDefaults<3>::set_truncate_on_project(truncate_on_project);
        FunctionDefaults<3>::set_apply_randomize(apply_randomize);
        FunctionDefaults<3>::set_project_randomize(project_randomize);
        FunctionDefaults<3>::set_cell(cell);
    }

  private:
    int k;
    double thresh;
    int initial_level;
    int truncate_mode;
    bool refine;
    bool autorefine;
    bool truncate_on_project;
    bool apply_randomize;
    bool project_randomize;
    Tensor<double> cell;
};

std::vector<double> tensor_to_vector(const Tensor<double>& t) {
    std::vector<double> result;
    for (long i = 0; i < t.size(); ++i)
        result.push_back(t(i));
    return result;
}

std::vector<SavedFct<3>> to_saved_functions(const std::vector<real_function_3d>& functions, const std::string& info,
                                            std::size_t n) {
    std::vector<SavedFct<3>> result;
    for (std::size_t i = 0; i < std::min(n, functions.size()); ++i)
        result.push_back(SavedFct<3>(functions[i], info));
    return result;
}

} // namespace

HartreeFock::HartreeFock(MadnessProcess<3>& mp, const MolecularGeometry& molecule, const std::string& parameters_json)
    : madness_process(mp), molecule(molecule.mol) {
    if (!parameters_json.empty())
        param.from_json(nlohmann::json::parse(parameters_json));
}

void HartreeFock::set_initial_orbitals(const std::vector<SavedFct<3>>& alpha_orbitals,
                                       const std::vector<double>& alpha_energies,
                                       const std::vector<SavedFct<3>>& beta_orbitals,
                                       const std::vector<double>& beta_energies) {
    guess_alpha.clear();
    guess_beta.clear();
    for (const auto& orb : alpha_orbitals)
        guess_alpha.push_back(madness_process.loadfct(orb));
    for (const auto& orb : beta_orbitals)
        guess_beta.push_back(madness_process.loadfct(orb));
    guess_alpha_energies = alpha_energies;
    guess_beta_energies = beta_energies;
}

// installs the guess orbitals in the SCF object, mirrors what SCF::initial_guess_from_nwchem does
// but with orbitals that were already translated into MRA
void HartreeFock::install_initial_orbitals(World& world) {
    const int nalpha = calc->param.nalpha();
    const int nbeta = calc->param.nbeta();
    const bool unrestricted = !calc->param.spin_restricted() && nbeta > 0;

    if (guess_alpha.size() != std::size_t(calc->param.nmo_alpha()))
        throw std::invalid_argument("HartreeFock: got " + std::to_string(guess_alpha.size()) +
                                    " alpha guess orbitals, but the calculation has " + std::to_string(nalpha) +
                                    " occupied alpha orbitals");
    if (unrestricted && guess_beta.size() != std::size_t(calc->param.nmo_beta()))
        throw std::invalid_argument("HartreeFock: got " + std::to_string(guess_beta.size()) +
                                    " beta guess orbitals, but the calculation has " + std::to_string(nbeta) +
                                    " occupied beta orbitals");

    // bring the guess to the polynomial order and threshold of the current protocol step
    auto prepare = [&world](std::vector<real_function_3d> mos) {
        const int k = FunctionDefaults<3>::get_k();
        const double thresh = FunctionDefaults<3>::get_thresh();
        reconstruct(world, mos);
        for (auto& mo : mos) {
            if (mo.k() != k)
                mo = madness::project(mo, k, thresh, false);
        }
        world.gop.fence();
        truncate(world, mos);
        normalize(world, mos);
        return mos;
    };
    auto energies_tensor = [](const std::vector<double>& energies, std::size_t n) {
        tensorT eps(n);
        for (std::size_t i = 0; i < std::min(n, energies.size()); ++i)
            eps(i) = energies[i];
        return eps;
    };
    auto occupations_tensor = [](std::size_t n) {
        tensorT occ(n);
        occ.fill(1.0);
        return occ;
    };

    calc->amo = prepare(guess_alpha);
    calc->aocc = occupations_tensor(guess_alpha.size());
    calc->aeps = energies_tensor(guess_alpha_energies, guess_alpha.size());
    calc->aset = calc->group_orbital_sets(world, calc->aeps, calc->aocc, calc->param.nmo_alpha());

    if (unrestricted) {
        calc->bmo = prepare(guess_beta);
        calc->bocc = occupations_tensor(guess_beta.size());
        calc->beps = energies_tensor(guess_beta_energies, guess_beta.size());
        calc->bset = calc->group_orbital_sets(world, calc->beps, calc->bocc, calc->param.nmo_beta());
    }
}

double HartreeFock::solve() {
    World& world = *(madness_process.world);
    FunctionDefaultsGuard guard;

    calc = std::make_shared<SCF>(world, param, molecule);
    const std::vector<double> protocol = calc->param.protocol();
    if (protocol.empty())
        throw std::invalid_argument("HartreeFock: the protocol must contain at least one threshold");

    calc->set_protocol<3>(world, protocol[0]);
    calc->make_nuclear_potential(world);
    if (guess_alpha.empty()) {
        // default MADNESS guess: diagonalization of the Fock matrix in the AO basis param.aobasis()
        calc->reset_aobasis(calc->param.aobasis());
        calc->ao = calc->project_ao_basis(world, calc->aobasis);
        calc->initial_guess(world);
    } else {
        install_initial_orbitals(world);
    }

    // the sto-3g basis is used by the SCF for the analysis of the orbitals and for localization
    calc->reset_aobasis("sto-3g");
    for (std::size_t i = 0; i < protocol.size(); ++i) {
        calc->set_protocol<3>(world, protocol[i]);
        calc->make_nuclear_potential(world);
        if (i > 0)
            calc->project(world);
        calc->ao.clear();
        world.gop.fence();
        calc->ao = calc->project_ao_basis(world, calc->aobasis);
        calc->solve(world);
        if (calc->param.save())
            calc->save_mos(world);
    }
    return calc->current_energy;
}

void HartreeFock::check_solved() const {
    if (!calc)
        throw std::runtime_error("HartreeFock: no results available, call solve() first");
}

std::vector<SavedFct<3>> HartreeFock::get_alpha_orbitals() const {
    check_solved();
    return to_saved_functions(calc->amo, "hf_alpha", calc->amo.size());
}

std::vector<SavedFct<3>> HartreeFock::get_beta_orbitals() const {
    check_solved();
    // spin restricted: the beta orbitals are the lowest nbeta alpha orbitals
    if (calc->param.spin_restricted())
        return to_saved_functions(calc->amo, "hf_beta", calc->param.nbeta());
    return to_saved_functions(calc->bmo, "hf_beta", calc->bmo.size());
}

std::vector<double> HartreeFock::get_alpha_orbital_energies() const {
    check_solved();
    return tensor_to_vector(calc->aeps);
}

std::vector<double> HartreeFock::get_beta_orbital_energies() const {
    check_solved();
    if (calc->param.spin_restricted()) {
        std::vector<double> eps = tensor_to_vector(calc->aeps);
        eps.resize(std::min<std::size_t>(eps.size(), calc->param.nbeta()));
        return eps;
    }
    return tensor_to_vector(calc->beps);
}

std::vector<SavedFct<3>> HartreeFock::get_minimal_basis() const {
    check_solved();
    return to_saved_functions(calc->ao, "sto-3g", calc->ao.size());
}

double HartreeFock::get_energy() const {
    check_solved();
    return calc->current_energy;
}

SavedFct<3> HartreeFock::get_vnuc() const {
    check_solved();
    return SavedFct<3>(calc->potentialmanager->vnuclear());
}

double HartreeFock::get_nuclear_repulsion() const {
    return molecule.nuclear_repulsion_energy();
}
