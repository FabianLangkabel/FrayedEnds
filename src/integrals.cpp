#include "integrals.hpp"

using namespace madness;

namespace {

std::size_t symmetric_pair_index(std::size_t i, std::size_t j) {
    return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}

std::size_t symmetric_pair_count(std::size_t dimension) {
    return dimension * (dimension + 1) / 2;
}

} // namespace

template <std::size_t NDIM> Integrals<NDIM>::Integrals(MadnessProcess<NDIM>& mp) : madness_process(mp) {}

//
// Utility functions
//

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::read_orbitals(std::vector<SavedFct<NDIM>> orbs) {
    std::vector<Function<double, NDIM>> orbitals;
    for (SavedFct<NDIM> orb : orbs)
        orbitals.push_back(madness_process.loadfct(orb));
    return orbitals;
}

template <std::size_t NDIM>
void Integrals<NDIM>::update_as_integral_combinations(const std::vector<Function<double, NDIM>> &orbitals, std::vector<Function<double, NDIM>> &orbs_kl, std::vector<Function<double, NDIM>> &coul_orbs_mn) {
    // Precompute active orbital combinations. Store only k >= l. Ordering according to: (k,l) is stored at max(k,l)*(max(k,l)+1)/2+min(k,l).
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));

    orbs_kl.clear();
    coul_orbs_mn.clear();

    const std::size_t n_unique_pairs = symmetric_pair_count(orbitals.size());
    orbs_kl.reserve(n_unique_pairs);
    std::vector<Function<double, NDIM>> l_orbs;
    for (int k = 0; k < orbitals.size(); k++) {
        l_orbs.push_back(orbitals[k]);
        std::vector<Function<double, NDIM>> kl = orbitals[k] * l_orbs;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, num_params.truncation_tol);
    coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, num_params.truncation_tol);
}
    
template <std::size_t NDIM>
void Integrals<NDIM>::update_core_integral_combinations(const std::vector<Function<double, NDIM>> &core_orbitals, std::vector<Function<double, NDIM>> &orbs_aa) {
    // Precompute the combinations of core orbitals, only aa combinations are saved because ab combinations take up too much memory for many orbitals
    orbs_aa.clear();

    for (int a = 0; a < core_orbitals.size(); a++) {
        orbs_aa.push_back(core_orbitals[a] * core_orbitals[a]);
    }
    orbs_aa = truncate(orbs_aa, num_params.truncation_tol);
}

template <std::size_t NDIM>
void Integrals<NDIM>::update_core_integral_combinations(const std::vector<Function<double, NDIM>> &core_orbitals, std::vector<Function<double, NDIM>> &orbs_aa, std::vector<Function<double, NDIM>> &coul_orbs_aa) {
    // Precompute orbs_aa and coul_orbs_aa as well
    orbs_aa.clear();
    coul_orbs_aa.clear();

    for (int a = 0; a < core_orbitals.size(); a++) {
        orbs_aa.push_back(core_orbitals[a] * core_orbitals[a]);
    }
    orbs_aa = truncate(orbs_aa, num_params.truncation_tol);

    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));
    coul_orbs_aa = apply(*(madness_process.world), *coul_op_parallel, orbs_aa);
    coul_orbs_aa = truncate(coul_orbs_aa, num_params.truncation_tol);
}

//
// Nanobind bindings
//

template <std::size_t NDIM>
Numpy2D Integrals<NDIM>::nb_compute_overlap_integrals(const std::vector<SavedFct<NDIM>>& all_orbs, const std::vector<SavedFct<NDIM>>& other) {
    std::vector<Function<double, NDIM>> orbitals1 = read_orbitals(all_orbs);
    std::vector<Function<double, NDIM>> orbitals2 = read_orbitals(other);
    
    Tensor<double>* integrals_pointer = new Tensor<double>(matrix_inner(*(madness_process.world), orbitals1, orbitals2));

    nb::capsule ints_capsule(
        integrals_pointer,
        [](void *p) noexcept {
            delete reinterpret_cast<Tensor<double>*>(p);
        }
    );

    return Numpy2D(integrals_pointer->ptr(), {orbitals1.size(), orbitals2.size()}, ints_capsule);
}

template <std::size_t NDIM>
Numpy2D Integrals<NDIM>::nb_compute_potential_integrals(const std::vector<SavedFct<NDIM>>& all_orbs, const SavedFct<NDIM>& potential) {
    std::vector<Function<double, NDIM>> orbitals = read_orbitals(all_orbs);
    Function<double, NDIM> V = madness_process.loadfct(potential);

    Tensor<double>* integrals_pointer = new Tensor<double>(compute_potential_integrals(orbitals, V));

    nb::capsule ints_capsule(
        integrals_pointer,
        [](void *p) noexcept {
            delete reinterpret_cast<Tensor<double>*>(p);
        }
    );

    return Numpy2D(integrals_pointer->ptr(), {orbitals.size(), orbitals.size()}, ints_capsule);
}

template <std::size_t NDIM>
Numpy2D Integrals<NDIM>::nb_compute_kinetic_integrals(const std::vector<SavedFct<NDIM>>& all_orbs) {
    std::vector<Function<double, NDIM>> orbitals = read_orbitals(all_orbs);

    Tensor<double>* integrals_pointer = new Tensor<double>(compute_kinetic_integrals(orbitals));

    nb::capsule ints_capsule(
        integrals_pointer,
        [](void *p) noexcept {
            delete reinterpret_cast<Tensor<double>*>(p);
        }
    );

    const size_t dim = orbitals.size();
    return Numpy2D(integrals_pointer->ptr(), {dim, dim}, ints_capsule);
}

template <std::size_t NDIM>
Numpy4D Integrals<NDIM>::nb_compute_two_body_integrals(const std::vector<SavedFct<NDIM>>& all_orbs) {
    std::vector<Function<double, NDIM>> orbitals = read_orbitals(all_orbs);

    std::vector<Function<double, NDIM>> orbs_kl;
    std::vector<Function<double, NDIM>> coul_orbs_mn;
    update_as_integral_combinations(orbitals, orbs_kl, coul_orbs_mn);

    Tensor<double>* integrals_pointer = new Tensor<double>(compute_two_body_integrals(orbitals, orbs_kl, coul_orbs_mn));

    nb::capsule ints_capsule(
        integrals_pointer,
        [](void *p) noexcept {
            delete reinterpret_cast<Tensor<double>*>(p);
        }
    );

    const size_t dim = orbitals.size();
    return Numpy4D(integrals_pointer->ptr(), {dim, dim, dim, dim}, ints_capsule);
}

template <std::size_t NDIM>
Numpy2D Integrals<NDIM>::nb_compute_frozen_core_interaction(const std::vector<SavedFct<NDIM>>& fr_c_orbs,
                                                           const std::vector<SavedFct<NDIM>>& a_orbs) {
    std::vector<Function<double, NDIM>> core_orbs = read_orbitals(fr_c_orbs);
    std::vector<Function<double, NDIM>> active_orbs = read_orbitals(a_orbs);

    const size_t core_dim = core_orbs.size();
    const size_t as_dim = active_orbs.size();

    std::vector<Function<double, NDIM>> orbs_kl;
    std::vector<Function<double, NDIM>> coul_orbs_mn;
    update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);

    std::vector<Function<double, NDIM>> orbs_aa;
    update_core_integral_combinations(core_orbs, orbs_aa);

    auto [core_as_integrals_two_body_akal, core_as_integrals_two_body_akla] = compute_core_as_2e_integrals_energy(core_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa);
    
    Tensor<double> result(as_dim, as_dim);
    for (int a = 0; a < core_dim; a++) {
            result += 2*core_as_integrals_two_body_akal(a, _, _) - core_as_integrals_two_body_akla(a, _, _);
    }

    Tensor<double>* integrals_pointer = new Tensor<double>(std::move(result));
    
    nb::capsule ints_capsule(
        integrals_pointer,
        [](void *p) noexcept {
            delete reinterpret_cast<Tensor<double>*>(p);
        }
    );

    return Numpy2D(integrals_pointer->ptr(), {as_dim, as_dim}, ints_capsule);
}

template <std::size_t NDIM>
nb::tuple Integrals<NDIM>::nb_compute_effective_hamiltonian(const std::vector<SavedFct<NDIM>>& core_orbitals, const std::vector<SavedFct<NDIM>>& active_orbitals, const SavedFct<NDIM>& potential, double energy_offset)
{
    std::vector<Function<double, NDIM>> core_orbs = read_orbitals(core_orbitals);
    std::vector<Function<double, NDIM>> active_orbs = read_orbitals(active_orbitals);
    Function<double, NDIM> V = madness_process.loadfct(potential);

    const size_t core_dim = core_orbs.size();
    const size_t as_dim = active_orbs.size();

    std::vector<Function<double, NDIM>> orbs_kl;
    std::vector<Function<double, NDIM>> coul_orbs_mn;
    update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);

    // Active space integrals
    Tensor<double> one_e_integrals = compute_potential_integrals(active_orbs, V);
    one_e_integrals += compute_kinetic_integrals(active_orbs);

    Tensor<double> two_e_integrals = compute_two_body_integrals(active_orbs, orbs_kl, coul_orbs_mn);

    // Core interactions
    double effective_hamiltonian_core_energy = energy_offset;
    if(core_dim > 0) 
    {
        std::vector<Function<double, NDIM>> orbs_aa;
        std::vector<Function<double, NDIM>> coul_orbs_aa;
        update_core_integral_combinations(core_orbs, orbs_aa, coul_orbs_aa);

        // Core energy
        effective_hamiltonian_core_energy = compute_core_energy(core_orbs, orbs_aa, coul_orbs_aa, V, energy_offset);

        // Core-AS interaction
        auto [core_as_integrals_two_body_akal, core_as_integrals_two_body_akla] = compute_core_as_2e_integrals_energy(core_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa);
        for (int a = 0; a < core_dim; a++) {
            one_e_integrals += 2*core_as_integrals_two_body_akal(a, _, _) - core_as_integrals_two_body_akla(a, _, _);
        }
    }

    Tensor<double>* one_e_pointer = new Tensor<double>(std::move(one_e_integrals));
    Tensor<double>* two_e_pointer = new Tensor<double>(std::move(two_e_integrals));
    nb::capsule one_e_capsule(
        one_e_pointer,
        [](void *p) noexcept { delete reinterpret_cast<Tensor<double>*>(p); }
    );
    nb::capsule two_e_capsule(
        two_e_pointer,
        [](void *p) noexcept { delete reinterpret_cast<Tensor<double>*>(p); }
    );

    return nb::make_tuple(
        effective_hamiltonian_core_energy,
        Numpy2D(one_e_pointer->ptr(), {as_dim, as_dim}, one_e_capsule),
        Numpy4D(two_e_pointer->ptr(), {as_dim, as_dim, as_dim, as_dim}, two_e_capsule));
}

//
// Integrators
//

template <std::size_t NDIM>
Tensor<double> Integrals<NDIM>::compute_potential_integrals(const std::vector<Function<double, NDIM>>& orbitals, const Function<double, NDIM>& V){
    // v(i,j) = <i|V|j>
    madness::Tensor<double> pot_ints;
    pot_ints = madness::matrix_inner(*(madness_process.world), orbitals, V * orbitals);
    return pot_ints;
}

template <std::size_t NDIM>
Tensor<double> Integrals<NDIM>::compute_kinetic_integrals(const std::vector<Function<double, NDIM>>& orbitals){
    // t(i,j) = -0.5*<i|Laplacian|j> = 0.5*sum_n <dx_n(i)|dx_n(j)>
    madness::Tensor<double> kin_ints = madness::Tensor<double>(orbitals.size(), orbitals.size());
    for (int axis = 0; axis < NDIM; axis++) {
        Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
        std::vector<Function<double, NDIM>> derivatives =
            apply(*(madness_process.world), D, orbitals);
        kin_ints += 0.5 * matrix_inner(*(madness_process.world), derivatives, derivatives, true);
    }
    return kin_ints;
}

template <std::size_t NDIM>
Tensor<double> Integrals<NDIM>::compute_two_body_integrals(const std::vector<Function<double, NDIM>> &orbitals, const std::vector<Function<double, NDIM>> &orbs_kl, const std::vector<Function<double, NDIM>> &coul_orbs_mn){
    // g(i,j,k,l) = <ij|g|kl> (physicist's notation)
    // orbs_kl and coul_orbs_mn may use compact triangular storage, or full square storage when the pair family is not symmetric.
    const std::size_t dimension = orbitals.size();
    const std::size_t pair_count = symmetric_pair_count(dimension);
    const bool compact_bra = orbs_kl.size() == pair_count; // check if orbs_kl is in compact triangular storage
    const bool compact_ket = coul_orbs_mn.size() == pair_count;
    MADNESS_CHECK_THROW(compact_bra || orbs_kl.size() == dimension * dimension, "Two-body bra pairs must use compact triangular or full square storage");
    MADNESS_CHECK_THROW(compact_ket || coul_orbs_mn.size() == dimension * dimension, "Two-body ket pairs must use compact triangular or full square storage");

    madness::Tensor<double> twob_ints = madness::Tensor<double>(dimension, dimension, dimension, dimension);
    madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);

    for (std::size_t k = 0; k < dimension; k++) {
        for (std::size_t l = 0; l < dimension; l++) {
            for (std::size_t m = 0; m < dimension; m++) {
                for (std::size_t n = 0; n < dimension; n++) {
                    const std::size_t bra_index = compact_bra ? symmetric_pair_index(k, l) : k * dimension + l;
                    const std::size_t ket_index = compact_ket ? symmetric_pair_index(m, n) : m * dimension + n;
                    twob_ints(k, m, l, n) = Inner_prods(bra_index, ket_index);
                }
            }
        }
    }
    return twob_ints;
}

template <std::size_t NDIM>
double Integrals<NDIM>::compute_core_energy(const std::vector<Function<double, NDIM>>& core_orbitals, std::vector<Function<double, NDIM>>& orbs_aa, std::vector<Function<double, NDIM>>& coul_orbs_aa, const Function<double, NDIM>& V, double energy_offset){
    // 1e core energy 
    double core_kinetic_energy = 0;
    for (int axis = 0; axis < NDIM; axis++) {
        Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
        std::vector<Function<double, NDIM>> derivatives = apply(*(madness_process.world), D, core_orbitals);
        core_kinetic_energy += inner(*(madness_process.world), derivatives, derivatives).sum();
    }
    // E_pot=2*\sum_a <a|V|a>
    std::vector<Function<double, NDIM>> potential_orbitals = V * core_orbitals;
    double core_nuclear_attraction_energy = 2 * inner(*(madness_process.world), core_orbitals, potential_orbitals).sum();

    // E_2e=\sum_ab 2<ab|ab>-<ab|ba>
    double core_two_electron_energy = 0;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));

    // check if orbs_aa is empty, if yes: build orbs_aa and coul_orbs_aa
    if (orbs_aa.size()==0) {update_core_integral_combinations(core_orbitals, orbs_aa, coul_orbs_aa);}

    // check that coul_orbs_aa is not empty, if yes: construct from orbs_aa
    if (coul_orbs_aa.size()==0) {
        coul_orbs_aa = apply(*(madness_process.world), *coul_op_parallel, orbs_aa);
        coul_orbs_aa = truncate(coul_orbs_aa, num_params.truncation_tol);
    }
    
    // <ab|ab>
    madness::Tensor<double> Inner_prods_abab = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_aa, false);
    core_two_electron_energy += 2 * Inner_prods_abab.sum();

    // <ab|ba>. The diagonal (aa|aa) is already available above, and the
    // off-diagonal terms are symmetric under a <-> b.
    std::vector<Function<double, NDIM>> b_core_orbitals;
    b_core_orbitals.reserve(core_orbitals.size());
    for (std::size_t a = 0; a < core_orbitals.size(); a++) {
        core_two_electron_energy -= Inner_prods_abab(a, a);
        if (!b_core_orbitals.empty()) {
            std::vector<Function<double, NDIM>> orbs_ab = core_orbitals[a] * b_core_orbitals;
            orbs_ab = truncate(orbs_ab, num_params.truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ab = apply(*(madness_process.world), *coul_op_parallel, orbs_ab);
            coul_orbs_ab = truncate(coul_orbs_ab, num_params.truncation_tol);
            core_two_electron_energy -= 2 * inner(*(madness_process.world), orbs_ab, coul_orbs_ab).sum();
        }
        b_core_orbitals.push_back(core_orbitals[a]);
    }

    double core_energy = energy_offset + core_kinetic_energy + core_nuclear_attraction_energy + core_two_electron_energy;
    //print("     Initial core energy (energy offset) ", energy_offset);
    //print("                   Core - Kinetic energy ", core_kinetic_energy);
    //print("        Core - Nuclear attraction energy ", core_nuclear_attraction_energy);
    //print("              Core - Two-electron energy ", core_two_electron_energy);
    //print("                       Total core energy ", core_energy);

    return core_energy;
}

template <std::size_t NDIM>
madness::Tensor<double> Integrals<NDIM>::compute_core_as_integrals_one_body(const std::vector<Function<double, NDIM>>& core_orbitals, const std::vector<Function<double, NDIM>>& active_orbitals, const Function<double, NDIM>& V)
{
    madness::Tensor<double> ints;
    ints = madness::matrix_inner(*(madness_process.world), core_orbitals, V * active_orbitals);
    for (int axis = 0; axis < NDIM; axis++) {
        Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
        std::vector<Function<double, NDIM>> core_derivatives = apply(*(madness_process.world), D, core_orbitals);
        std::vector<Function<double, NDIM>> active_derivatives = apply(*(madness_process.world), D, active_orbitals);
        ints += 0.5 * matrix_inner(*(madness_process.world), core_derivatives, active_derivatives, false);
    }
    return ints;
}

template <std::size_t NDIM>
std::array<Tensor<double>, 2> Integrals<NDIM>::compute_core_as_2e_integrals_energy(
    const std::vector<Function<double, NDIM>> &core_orbitals, 
    const std::vector<Function<double, NDIM>> &active_orbitals, 
    const std::vector<Function<double, NDIM>> &orbs_kl, 
    const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
    const std::vector<Function<double, NDIM>> &orbs_aa
) 
{
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));
    const std::size_t pair_count = symmetric_pair_count(active_orbitals.size());
    MADNESS_ASSERT(orbs_kl.size() == pair_count);
    MADNESS_ASSERT(coul_orbs_mn.size() == pair_count);

    Tensor<double> core_as_integrals_two_body_akal(core_orbitals.size(), active_orbitals.size(), active_orbitals.size());
    Tensor<double> core_as_integrals_two_body_akla(core_orbitals.size(), active_orbitals.size(), active_orbitals.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    //akal
    Tensor<double> Inner_prods_akal = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_mn, false);

    for (int a = 0; a < core_orbitals.size(); a++) {
        for (int k = 0; k < active_orbitals.size(); k++) {
            for (int l = 0; l < active_orbitals.size(); l++) {
                core_as_integrals_two_body_akal(a,k,l) = Inner_prods_akal(a, symmetric_pair_index(k, l));
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    //akla
    for (int a = 0; a < core_orbitals.size(); a++) {// One core orbital after the other -> Slightly less efficient than all a at
                                                    // the same time, but reduces memory
        std::vector<Function<double, NDIM>> orbs_ak = core_orbitals[a] * active_orbitals;
        orbs_ak = truncate(orbs_ak, num_params.truncation_tol);
        std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
        coul_orbs_ak = truncate(coul_orbs_ak, num_params.truncation_tol);

        // <ak|la> = <ka|al>
        Tensor<double> Inner_prods_akla = matrix_inner(*(madness_process.world), orbs_ak, coul_orbs_ak, false);
        for (int k = 0; k < active_orbitals.size(); k++) {
            for (int l = 0; l < active_orbitals.size(); l++) {
                core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "akal: " << std::chrono::duration<double>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "akla: " << std::chrono::duration<double>(t3 - t2).count() << " seconds" << std::endl;

    return std::array<Tensor<double>, 2>{core_as_integrals_two_body_akal, core_as_integrals_two_body_akla};

}


template <std::size_t NDIM>
std::array<Tensor<double>, 5> Integrals<NDIM>::compute_core_as_2e_integrals_as_refinement(
    const std::vector<Function<double, NDIM>> &core_orbitals, 
    const std::vector<Function<double, NDIM>> &active_orbitals, 
    const std::vector<Function<double, NDIM>> &orbs_kl, 
    const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
    const std::vector<Function<double, NDIM>> &orbs_aa,
    std::vector<Function<double, NDIM>> &sum_a_aka
)
{
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));
    const std::size_t pair_count = symmetric_pair_count(active_orbitals.size());
    MADNESS_ASSERT(orbs_kl.size() == pair_count);
    MADNESS_ASSERT(coul_orbs_mn.size() == pair_count);

    Tensor<double> core_as_integrals_two_body_akal(core_orbitals.size(), active_orbitals.size(), active_orbitals.size());
    Tensor<double> core_as_integrals_two_body_akla(core_orbitals.size(), active_orbitals.size(), active_orbitals.size());
    Tensor<double> core_as_integrals_two_body_akln(core_orbitals.size(), active_orbitals.size(), active_orbitals.size(), active_orbitals.size()); //stored as (a,k,l,n)
    Tensor<double> core_as_integrals_two_body_abak(core_orbitals.size(), core_orbitals.size(), active_orbitals.size()); //stored as (a,b,k)
    Tensor<double> core_as_integrals_two_body_baak(core_orbitals.size(), core_orbitals.size(), active_orbitals.size()); //stored as (a,b,k)
    sum_a_aka.clear();
    
    auto t1 = std::chrono::high_resolution_clock::now();

    //akal
    Tensor<double> Inner_prods_akal = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_mn, false);
    for (int a = 0; a < core_orbitals.size(); a++) {
        for (int k = 0; k < active_orbitals.size(); k++) {
            for (int l = 0; l < active_orbitals.size(); l++) {
                core_as_integrals_two_body_akal(a,k,l) = Inner_prods_akal(a, symmetric_pair_index(k, l));
            }
        }
    }

    for (int a = 0; a < core_orbitals.size(); a++)
    {   
        std::vector<Function<double, NDIM>> orbs_ak = core_orbitals[a] * active_orbitals;
        orbs_ak = truncate(orbs_ak, num_params.truncation_tol);
        std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
        coul_orbs_ak = truncate(coul_orbs_ak, num_params.truncation_tol);
        
        // <ak|la> = <ka|al>
        Tensor<double> Inner_prods_akla = matrix_inner(*(madness_process.world), orbs_ak, coul_orbs_ak, false);
        for (int k = 0; k < active_orbitals.size(); k++) {
            for (int l = 0; l < active_orbitals.size(); l++) {
                core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
            }
        }

        // <ak|ln>
        Tensor<double> Inner_prods_akln = matrix_inner(*(madness_process.world), orbs_ak, coul_orbs_mn, false);
        for (int k = 0; k < active_orbitals.size(); k++) {
            for (int l = 0; l < active_orbitals.size(); l++) {
                for (int n = 0; n < active_orbitals.size(); n++) {
                    core_as_integrals_two_body_akln(a, l, k, n) = Inner_prods_akln(k, symmetric_pair_index(l, n));
                }
            }
        }

        // calculate <ba|bk> and transform to <ab|ak>
        Tensor<double> Inner_prods_babk = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_ak, false); //orbs_aa are orbs_bb in this case (aa is independant of variable "a")
        for (int b = 0; b < core_orbitals.size(); b++) {
            for (int k = 0; k < active_orbitals.size(); k++) {
                core_as_integrals_two_body_abak(b,a,k) = Inner_prods_babk(b, k);
            }
        }

        // <ba|ak>, and retain only the contracted exchange action
        // sum_a phi_a C(phi_a phi_k), not the full core-active pair family.
        std::vector<Function<double, NDIM>> aka = core_orbitals[a] * coul_orbs_ak;
        aka = truncate(aka, num_params.truncation_tol);
        madness::Tensor<double> Inner_prods_baak = matrix_inner(*(madness_process.world), core_orbitals, aka, false);
        for (int b = 0; b < core_orbitals.size(); b++) {
            for (int k = 0; k < active_orbitals.size(); k++) {
                core_as_integrals_two_body_baak(a,b,k) = Inner_prods_baak(b, k);
            }
        }
        if (sum_a_aka.empty()) {
            sum_a_aka = std::move(aka);
        }
        else {
            sum_a_aka += aka;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "core_as_2e_as_refinement: " << std::chrono::duration<double>(t2 - t1).count() << " seconds" << std::endl;
    return std::array<Tensor<double>, 5>{core_as_integrals_two_body_akal, core_as_integrals_two_body_akla, core_as_integrals_two_body_akln, core_as_integrals_two_body_abak, core_as_integrals_two_body_baak};
}


template <std::size_t NDIM>
std::array<Tensor<double>, 4> Integrals<NDIM>::compute_core_as_2e_integrals_core_refinement(
    const std::vector<Function<double, NDIM>> &core_orbitals, 
    const std::vector<Function<double, NDIM>> &active_orbitals, 
    const std::vector<Function<double, NDIM>> &orbs_kl, 
    const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
    const std::vector<Function<double, NDIM>> &orbs_aa,
    const std::vector<Function<double, NDIM>> &coul_orbs_aa,
    std::vector<Function<double, NDIM>> &sum_a_aca
)
{
    const std::size_t pair_count = symmetric_pair_count(active_orbitals.size());
    MADNESS_ASSERT(orbs_kl.size() == pair_count);
    MADNESS_ASSERT(coul_orbs_mn.size() == pair_count);
    MADNESS_ASSERT(orbs_aa.size() == core_orbitals.size());
    MADNESS_ASSERT(coul_orbs_aa.size() == core_orbitals.size());

    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));

    Tensor<double> sum_a_core_as_integrals_two_body_baca(core_orbitals.size(), core_orbitals.size()); //stored as (b,c)
    Tensor<double> sum_a_core_as_integrals_two_body_baac(core_orbitals.size(), core_orbitals.size()); //stored as (b,c)
    Tensor<double> core_as_integrals_two_body_akcl(core_orbitals.size(), active_orbitals.size(), core_orbitals.size(), active_orbitals.size()); //stored as (a,k,c,l)
    Tensor<double> core_as_integrals_two_body_aklc(core_orbitals.size(), active_orbitals.size(), active_orbitals.size(), core_orbitals.size()); //stored as (a,k,l,c)

    auto t1 = std::chrono::high_resolution_clock::now();
    
    Function<double, NDIM> summed_coul_orbs_aa = sum(*(madness_process.world), coul_orbs_aa);
    sum_a_aca = mul(*(madness_process.world), core_orbitals, coul_orbs_aa); // add diagonal terms (phi_a*C(phi_a*phi_a)) to sum_a_aca
    sum_a_aca = truncate(sum_a_aca, num_params.truncation_tol);

    for (int b = 0; b < core_orbitals.size(); b++)
    {   
        std::vector<Function<double, NDIM>> c_core_orbitals(core_orbitals.begin() + b + 1, core_orbitals.end()); // all core orbitals with index c > b

        std::vector<Function<double, NDIM>> orbs_bc{orbs_aa[b]};
        std::vector<Function<double, NDIM>> coul_orbs_bc{coul_orbs_aa[b]};
        std::vector<Function<double, NDIM>> coul_orbs_bc_rest;

        if (!c_core_orbitals.empty()) {
            std::vector<Function<double, NDIM>> orbs_bc_rest = core_orbitals[b] * c_core_orbitals;
            orbs_bc_rest = truncate(orbs_bc_rest, num_params.truncation_tol);
            orbs_bc.insert(orbs_bc.end(), orbs_bc_rest.begin(), orbs_bc_rest.end());

            coul_orbs_bc_rest = apply(*(madness_process.world), *coul_op_parallel, orbs_bc_rest);
            coul_orbs_bc_rest = truncate(coul_orbs_bc_rest, num_params.truncation_tol);
            coul_orbs_bc.insert(coul_orbs_bc.end(), coul_orbs_bc_rest.begin(), coul_orbs_bc_rest.end());
        }

        // \sum_a <ba|ca>
        madness::Tensor<double> sum_a_inner_prod_baca = matrix_inner(*(madness_process.world), orbs_bc, std::vector<Function<double,NDIM>>({summed_coul_orbs_aa}), false);
        for (std::size_t pair_offset = 0; pair_offset < orbs_bc.size(); pair_offset++) {
            const std::size_t c = b + pair_offset;
            sum_a_core_as_integrals_two_body_baca(b, c) = sum_a_inner_prod_baca(pair_offset, 0);
            sum_a_core_as_integrals_two_body_baca(c, b) = sum_a_inner_prod_baca(pair_offset, 0);
        }

        // calculate <bk|cl> which is the same as <ak|cl>
        madness::Tensor<double> Inner_prod_bkcl = matrix_inner(*(madness_process.world), orbs_bc, coul_orbs_mn, false);
        for (std::size_t pair_offset = 0; pair_offset < orbs_bc.size(); pair_offset++) {
            for (int k = 0; k < active_orbitals.size(); k++) {
                for (int l = 0; l < active_orbitals.size(); l++) {
                    const std::size_t c = b + pair_offset;
                    core_as_integrals_two_body_akcl(b, k, c, l) = Inner_prod_bkcl(pair_offset, symmetric_pair_index(k, l));
                    core_as_integrals_two_body_akcl(c, k, b, l) = Inner_prod_bkcl(pair_offset, symmetric_pair_index(k, l));
                }
            }
        }
        
        if (!c_core_orbitals.empty()) {
            std::vector<Function<double, NDIM>> c_coul_bc = mul(*(madness_process.world), c_core_orbitals, coul_orbs_bc_rest);
            c_coul_bc = truncate(c_coul_bc, num_params.truncation_tol);
            sum_a_aca[b] += sum(*(madness_process.world), c_coul_bc);

            std::vector<Function<double, NDIM>> b_coul_cb = core_orbitals[b] * coul_orbs_bc_rest; 
            b_coul_cb = truncate(b_coul_cb, num_params.truncation_tol);
            for (std::size_t pair_offset = 0; pair_offset < b_coul_cb.size(); pair_offset++) {
                sum_a_aca[b + 1 + pair_offset] += b_coul_cb[pair_offset];
            }
        }        
    }

    sum_a_core_as_integrals_two_body_baac = matrix_inner(*(madness_process.world), core_orbitals, sum_a_aca, false);

    // <ak|lc> 
    for (int c = 0; c < core_orbitals.size(); c++)
    {
        std::vector<Function<double, NDIM>> orbs_kc = active_orbitals * core_orbitals[c];
        orbs_kc = truncate(orbs_kc, num_params.truncation_tol);
        std::vector<Function<double, NDIM>> coul_orbs_kc = apply(*(madness_process.world), *coul_op_parallel, orbs_kc);
        coul_orbs_kc = truncate(coul_orbs_kc, num_params.truncation_tol);
        
        for (int a = 0; a < core_orbitals.size(); a++)
        {
            std::vector<Function<double, NDIM>> orbs_al = core_orbitals[a] * active_orbitals;
            orbs_al = truncate(orbs_al, num_params.truncation_tol);

            madness::Tensor<double> Inner_prod_aklc = matrix_inner(*(madness_process.world), orbs_al, coul_orbs_kc, false);
            for (int k = 0; k < active_orbitals.size(); k++) {
                for (int l = 0; l < active_orbitals.size(); l++) {
                    core_as_integrals_two_body_aklc(a, k, l, c) = Inner_prod_aklc(l, k);
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "core_as_2e_core_refinement: " << std::chrono::duration<double>(t2 - t1).count() << " seconds" << std::endl;

    return std::array<Tensor<double>, 4>{sum_a_core_as_integrals_two_body_baca, sum_a_core_as_integrals_two_body_baac, core_as_integrals_two_body_akcl, core_as_integrals_two_body_aklc};
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::normalize(std::vector<SavedFct<NDIM>> all_orbs) {

    std::vector<Function<double, NDIM>> basis;
    for (SavedFct<NDIM> orb : all_orbs)
        basis.push_back(madness_process.loadfct(orb));

    madness::normalize(*(madness_process.world), basis);

    std::vector<SavedFct<NDIM>> result;
    for (auto x : basis)
        result.push_back(SavedFct<NDIM>(x));
    for (size_t k = 0; k < result.size(); k++)
        result[k].info = all_orbs[k].info;

    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::orthonormalize(std::vector<SavedFct<NDIM>> all_orbs,
                                                            const std::string method, double rr_thresh,
                                                            nb::ndarray<nb::numpy, double, nb::ndim<1>> occupations_arr,
                                                            double degeneracy_tol) {
    std::vector<Function<double, NDIM>> basis;
    for (SavedFct<NDIM> orb : all_orbs)
        basis.push_back(madness_process.loadfct(orb));

    auto out_basis = basis;

    if (method == "mixed") {
        std::vector<double> occupations;

        for (size_t i = 0; i < occupations_arr.size(); i++) {
                occupations.push_back(occupations_arr(i));
        }

        if (occupations.size() != all_orbs.size()) {
            MADNESS_EXCEPTION("mixed orthonormalization: number of occupations must match number of orbitals", 1);
        }

        out_basis = orthonormalize_mixed_by_degeneracy(basis, occupations, degeneracy_tol);
    } else {
        auto S = madness::matrix_inner(*(madness_process.world), basis, basis, true);

        if (method == "cholesky") {
            out_basis = madness::orthonormalize_cd(basis, S);
        } else if (method == "symmetric") {
            out_basis = madness::orthonormalize_symmetric(basis, S);
        } else if (method == "canonical") {
            out_basis = madness::orthonormalize_canonical(basis, S, rr_thresh);
        } else if (method == "rr_cholesky") {
            out_basis = madness::orthonormalize_rrcd(basis, S, rr_thresh);
        } else {
            MADNESS_EXCEPTION("unknown orthonormalization method", 1);
        }
    }

    std::vector<SavedFct<NDIM>> result;
    for (auto x : out_basis)
        result.push_back(SavedFct<NDIM>(x));
    for (size_t k = 0; k < result.size(); k++)
        result[k].info = all_orbs[k].info;
    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::project_out(std::vector<SavedFct<NDIM>> kernel,
                                                         std::vector<SavedFct<NDIM>> target) {
    std::vector<Function<double, NDIM>> x;
    for (SavedFct<NDIM> orb : kernel)
        x.push_back(madness_process.loadfct(orb));
    std::vector<Function<double, NDIM>> y;
    for (SavedFct<NDIM> orb : target)
        y.push_back(madness_process.loadfct(orb));

    auto Q = madness::QProjector<double, NDIM>(x);
    auto z = Q(y);
    madness::normalize(*(madness_process.world), z);
    std::vector<SavedFct<NDIM>> result;
    for (size_t k = 0; k < target.size(); k++)
        result.push_back(SavedFct<NDIM>(z[k], target[k].info));
    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::project_on(std::vector<SavedFct<NDIM>> kernel,
                                                        std::vector<SavedFct<NDIM>> target) {
    std::vector<Function<double, NDIM>> x;
    for (SavedFct<NDIM> orb : kernel)
        x.push_back(madness_process.loadfct(orb));
    std::vector<Function<double, NDIM>> y;
    for (SavedFct<NDIM> orb : target)
        y.push_back(madness_process.loadfct(orb));

    auto P = madness::Projector<double, NDIM>(x);
    auto z = P(y);
    madness::normalize(*(madness_process.world), z);
    std::vector<SavedFct<NDIM>> result;
    for (size_t k = 0; k < target.size(); k++)
        result.push_back(SavedFct<NDIM>(z[k], target[k].info));
    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::transform(std::vector<SavedFct<NDIM>> orbitals, Numpy2D matrix) {
    std::vector<Function<double, NDIM>> x;
        for (SavedFct<NDIM> orb : orbitals)
            x.push_back(madness_process.loadfct(orb));

        // @todo there are more efficient ways (flatten and rewire the pointer of the first entry)
        madness::Tensor<double> U(matrix.shape(0), matrix.shape(1));
        for (auto k = 0; k < matrix.shape(0); ++k) {
            for (auto l = 0; l < matrix.shape(1); ++l) {
                U(k, l) = matrix(k, l);
            }
        }

        auto y = madness::transform(*(madness_process.world), x, U);

        std::vector<SavedFct<NDIM>> result;
        for (size_t k = 0; k < matrix.shape(1); k++)
            result.push_back(SavedFct<NDIM>(y[k], orbitals[k].info + " transformed "));
        return result;
}

template <std::size_t NDIM>
SavedFct<NDIM> Integrals<NDIM>::compute_electron_density(std::vector<SavedFct<NDIM>> core_orbitals, std::vector<SavedFct<NDIM>> active_orbitals, Numpy2D rdm1){
    // compute electron density from core and active orbitals and 1-RDM of active space
    std::vector<Function<double, NDIM>> core = read_orbitals(core_orbitals);
    std::vector<Function<double, NDIM>> active = read_orbitals(active_orbitals);
    auto rdm1_tensor = refinement_utils::to_madness(rdm1);

    int core_dim = core.size();
    int as_dim = active.size();

    auto ActiveSpaceRotationMatrix = madness::Tensor<double>(as_dim, as_dim);
    madness::Tensor<double> evals(as_dim);
    madness::syev(rdm1_tensor, ActiveSpaceRotationMatrix, evals);
    active = madness::transform(*(madness_process.world), active, ActiveSpaceRotationMatrix);

    Function<double, NDIM> density = madness::FunctionFactory<double, NDIM>(*(madness_process.world));
    for (int i = 0; i < as_dim; i++) density += evals[i] * active[i] * active[i];
    for (int i = 0; i < core_dim; i++) density += 2 * core[i] * core[i];

    return SavedFct<NDIM>(density);
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::orthonormalize_mixed_by_degeneracy(
    std::vector<Function<double, NDIM>>& orbitals,
    const std::vector<double>& occupations,
    double degeneracy_tol) {
    return refinement_utils::orthonormalize_mixed_by_degeneracy(
        *(madness_process.world), orbitals, occupations, degeneracy_tol);
}

template class Integrals<2>;
template class Integrals<3>;
