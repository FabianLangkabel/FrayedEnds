#include "hartree_fock.hpp"
#include <madness/chem/localizer.h>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace madness;

HartreeFock::HartreeFock(MadnessProcess<3>& mp, const MolecularGeometry& molecule, const std::string& parameters_json,
                         const std::string& localize_result)
    : madness_process(mp), molecule(molecule.mol), localize_result(localize_result) {
    if (!parameters_json.empty())
        param.from_json(nlohmann::json::parse(parameters_json));
    if (!localize_result.empty())
        CalculationParameters().set_user_defined_value("localize", localize_result); // just a check to see if method is known to MADNESS, never reused
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
    auto install = [&](const vecfuncT& guess, const std::vector<double>& energies, int n, const std::string& spin,
                       vecfuncT& mo, tensorT& occ, tensorT& eps, std::vector<int>& set) {
        if (guess.size() != std::size_t(n))
            throw std::invalid_argument("HartreeFock: got " + std::to_string(guess.size()) + " " + spin +
                                        " guess orbitals, but the calculation has " + std::to_string(n) + " occupied " +
                                        spin + " orbitals");
        // bring the guess to the polynomial order and threshold of the current protocol step
        mo = guess;
        reconstruct(world, mo);
        for (auto& f : mo)
            if (f.k() != FunctionDefaults<3>::get_k())
                f = madness::project(f, FunctionDefaults<3>::get_k(), FunctionDefaults<3>::get_thresh(), false);
        world.gop.fence();
        truncate(world, mo);
        normalize(world, mo);
        occ = tensorT(n).fill(1.0);
        eps = tensorT(n);
        std::copy_n(energies.begin(), std::min<std::size_t>(n, energies.size()), eps.ptr());
        set = calc->group_orbital_sets(world, eps, occ, n);
    };
    install(guess_alpha, guess_alpha_energies, calc->param.nalpha(), "alpha", calc->amo, calc->aocc, calc->aeps,
            calc->aset);
    if (!calc->param.spin_restricted() && calc->param.nbeta() > 0)
        install(guess_beta, guess_beta_energies, calc->param.nbeta(), "beta", calc->bmo, calc->bocc, calc->beps,
                calc->bset);
}

double HartreeFock::solve() {
    World& world = *madness_process.world;
    calc = std::make_shared<SCF>(world, param, molecule);
    if (calc->param.nmo_alpha() != calc->param.nalpha() || calc->param.nmo_beta() != calc->param.nbeta())
        throw std::invalid_argument("HartreeFock: virtual orbitals (nvalpha, nvbeta) are not supported");
    if (world.rank() == 0) {
        calc->param.print("dft", "end");
        molecule.print();
    }
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
    if (!localize_result.empty())
        rotate_final_orbitals(world);
    return calc->current_energy;
}

// Rotates the converged occupied orbitals once: canonical (localize_result="canon") or localized with a MADNESS
// localization method. Mirrors the last iteration of SCF::solve with param.localize set to localize_result: the Fock
// matrix of the converged orbitals is either diagonalized or transformed into the localized basis (orbital energies
// are then its diagonal elements). The density and the energy are invariant.
void HartreeFock::rotate_final_orbitals(World& world) {
    calc->param.set_user_defined_value("localize", localize_result);
    const bool unrestricted = !calc->param.spin_restricted() && calc->param.nbeta() > 0;
    real_function_3d rho = calc->make_density(world, calc->aocc, calc->amo);
    if (unrestricted)
        rho += calc->make_density(world, calc->bocc, calc->bmo);
    else if (calc->param.nbeta() > 0)
        rho.scale(2.0);
    real_function_3d vlocal = (calc->make_coulomb_potential(rho) + calc->potentialmanager->vnuclear()).truncate();

    auto rotate = [&](vecfuncT& mo, tensorT& eps, const tensorT& occ, int ispin) {
        double exc, enl, ekin;
        vecfuncT Vpsi = calc->apply_potential(world, occ, mo, vlocal, exc, enl, ispin);
        tensorT fock = calc->make_fock_matrix(world, mo, Vpsi, occ, ekin);
        tensorT U;
        if (calc->param.do_localize()) {
            Localizer localizer(world, calc->aobasis, molecule, calc->ao);
            localizer.set_method(calc->param.localize_method());
            MolecularOrbitals<double, 3> mos(mo, eps, {}, occ, calc->group_orbital_sets(world, eps, occ, mo.size()));
            U = transpose(localizer.compute_localization_matrix(world, mos, true));
            fock = inner(U, inner(fock, U, 1, 0), 0, 0);
            for (long i = 0; i < eps.size(); ++i)
                eps(i) = fock(i, i);
        } else {
            sygvp(world, fock, matrix_inner(world, mo, mo, true), 1, U, eps);
        }
        mo = transform(world, mo, U);
        truncate(world, mo);
        normalize(world, mo);
    };
    rotate(calc->amo, calc->aeps, calc->aocc, 0);
    if (unrestricted)
        rotate(calc->bmo, calc->beps, calc->bocc, 1);
    if (world.rank() == 0)
        print("\nfinal orbitals (localize_result =", localize_result, "), orbital energies\n", calc->aeps);
}

const SCF& HartreeFock::scf() const {
    if (!calc)
        throw std::runtime_error("HartreeFock: no results available, call solve() first");
    return *calc;
}

std::vector<SavedFct<3>> HartreeFock::to_saved(const vecfuncT& f, const std::string& info, std::size_t n) {
    std::vector<SavedFct<3>> result;
    for (std::size_t i = 0; i < std::min(n, f.size()); ++i)
        result.push_back(SavedFct<3>(f[i], info));
    return result;
}

std::vector<double> HartreeFock::to_vector(const Tensor<double>& t, std::size_t n) {
    const Tensor<double> c = copy(t); // contiguous storage
    return {c.ptr(), c.ptr() + std::min<std::size_t>(n, c.size())};
}

// spin restricted: the beta orbitals are the lowest nbeta alpha orbitals
std::vector<SavedFct<3>> HartreeFock::get_beta_orbitals() const {
    const bool restricted = scf().param.spin_restricted();
    return to_saved(restricted ? calc->amo : calc->bmo, "hf_beta", calc->param.nbeta());
}

std::vector<double> HartreeFock::get_beta_orbital_energies() const {
    const bool restricted = scf().param.spin_restricted();
    return to_vector(restricted ? calc->aeps : calc->beps, calc->param.nbeta());
}
