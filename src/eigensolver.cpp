#include "eigensolver.hpp"
#include "madness_process.hpp"
#include "functionsaver.hpp"

using namespace madness;

template <std::size_t NDIM>
Eigensolver<NDIM>::Eigensolver(MadnessProcess<NDIM>& mp) : madness_process(mp) {
    std::cout.precision(6);
}

template <std::size_t NDIM>
Eigensolver<NDIM>::~Eigensolver() {
    V.clear();
    orbitals.clear();
}

// Function to solve the eigenvalue problem for the given potential
template <std::size_t NDIM>
void Eigensolver<NDIM>::solve(SavedFct<NDIM> input_V, int num_levels, int max_iter) {
    Function<double, NDIM> V = madness_process.loadfct(input_V);
    std::cout << "Potential loaded" << std::endl;
    // Create the guess generator
    GuessGenerator<double, NDIM> guess_generator(*(madness_process.world)); // Guess generator for all potentials
    // Create the guess functions
    std::vector<Function<double, NDIM>> guesses = guess_generator.create_guesses(num_levels, V);

    // plot guess functions
    // for (int i = 0; i < guesses.size(); i++) {
    //    madness_process.plot("g-" + std::to_string(i) + ".dat", SavedFct<NDIM>(guesses[i]));
    //}

    // Diagonalize the Hamiltonian matrix
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(guesses, V);
    std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;

    // store the eigenfunctions in vector eigenfunctions
    std::vector<Function<double, NDIM>> eigenfunctions;

    // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
    for (int i = 0; i < num_levels; i++) {
        Function<double, NDIM> phi = optimize(V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
        eigenfunctions.push_back(phi);
    }

    // diagonalize the eigenfunctions again
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(eigenfunctions, V);
    std::cout << "Diagonalize" << std::endl;
    orbitals = diagonalized.second;
}

// Function to solve the eigenvalue problem for the given potential with given guesses
template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Eigensolver<NDIM>::solve_with_input_guesses(SavedFct<NDIM> input_V,
                                                                         const std::vector<SavedFct<NDIM>>& input_guesses,
                                                                         int num_levels, int max_iter) {
    Function<double, NDIM> V = madness_process.loadfct(input_V);
    std::vector<Function<double, NDIM>> guesses;

    for (const auto& guess : input_guesses) {
        guesses.push_back(madness_process.loadfct(guess));
    }

    // Diagonalize the Hamiltonian matrix
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(guesses, V);
    std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;

    // store the eigenfunctions in vector eigenfunctions
    std::vector<Function<double, NDIM>> eigenfunctions;

    // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
    for (int i = 0; i < num_levels; i++) {
        Function<double, NDIM> phi = optimize(V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
        eigenfunctions.push_back(phi);
    }

    // diagonalize the eigenfunctions again
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(eigenfunctions, V);
    std::cout << "Diagonalize" << std::endl;
    std::vector<Function<double, NDIM>> y = diagonalized.second;

    return y;
}

// Function to calculate the energy
template <std::size_t NDIM>
double Eigensolver<NDIM>::energy(const Function<double, NDIM>& phi, const Function<double, NDIM>& V) {
    double potential_energy = inner(phi, V * phi); // <phi|Vphi> = <phi|V|phi>
    double kinetic_energy = 0.0;

    for (int axis = 0; axis < NDIM; axis++) {
        Derivative<double, NDIM> D =
            free_space_derivative<double, NDIM>(*(madness_process.world), axis); // Derivative operator

        Function<double, NDIM> dphi = D(phi);
        kinetic_energy += 0.5 * inner(dphi, dphi); // (1/2) <dphi/dx | dphi/dx>
    }

    double energy = kinetic_energy + potential_energy;
    return energy;
}

// Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
template <std::size_t NDIM>
std::pair<Tensor<double>, std::vector<Function<double, NDIM>>>
Eigensolver<NDIM>::diagonalize(const std::vector<Function<double, NDIM>>& functions, const Function<double, NDIM>& V) {
    const int num = functions.size();

    auto H = Tensor<double>(num, num);           // Hamiltonian matrix
    auto overlap = Tensor<double>(num, num);     // Overlap matrix
    auto diag_matrix = Tensor<double>(num, num); // Diagonal matrix

    // Calculate the Hamiltonian matrix

    for (int i = 0; i < num; i++) {
        auto energy1 = energy(functions[i], V);
        std::cout << energy1 << std::endl;
        for (int j = 0; j < num; j++) {
            double kin_energy = 0.0;
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D =
                    free_space_derivative<double, NDIM>(*(madness_process.world), axis); // Derivative operator

                Function<double, NDIM> dx_i = D(functions[i]);
                Function<double, NDIM> dx_j = D(functions[j]);

                kin_energy += 0.5 * inner(dx_i, dx_j); // (1/2) <dphi/dx | dphi/dx>
            }

            double pot_energy = inner(functions[i], V * functions[j]); // <phi|V|phi>

            H(i, j) = kin_energy + pot_energy; // Hamiltonian matrix
        }
    }
    std::cout << "H: \n" << H << std::endl;

    // Calculate the Overlap matrix
    overlap = matrix_inner(*(madness_process.world), functions, functions);

    // Calculate the Diagonal matrix
    Tensor<double> U;
    Tensor<double> evals;
    // sygvp is a function to solve the generalized eigenvalue problem HU = SUW where S is the overlap matrix and W is
    // the diagonal matrix of eigenvalues of H The eigenvalues are stored in evals and the eigenvectors are stored in U
    sygvp(*(madness_process.world), H, overlap, 1, U, evals);

    diag_matrix.fill(0.0);
    for (int i = 0; i < num; i++) {
        diag_matrix(i, i) = evals(i); // Set the diagonal elements
    }

    std::cout << "dia_matrix: \n" << diag_matrix << std::endl;

    std::vector<Function<double, NDIM>> y;

    // y = U * functions
    y = transform(*(madness_process.world), functions, U);

    // std::cout << "U matrix: \n" << U << std::endl;
    // std::cout << "evals: \n" << evals << std::endl;
    return std::make_pair(evals, y);
}

// Function to optimize the eigenfunction for each energy level
template <std::size_t NDIM>
Function<double, NDIM> Eigensolver<NDIM>::optimize(Function<double, NDIM>& V, const Function<double, NDIM> guess_function, int N,
                                            const std::vector<Function<double, NDIM>>& prev_phi, int max_iter) {

    // Create the initial guess wave function
    Function<double, NDIM> phi = guess_function;

    phi.scale(1.0 / phi.norm2()); // phi *= 1.0/norm
    double E = energy(phi, V);

    NonlinearSolverND<NDIM> solver;
    int count_shift = 0; // counter how often the potential was shifted

    for (int iter = 0; iter <= max_iter; iter++) {

        // plot("phi-" + std::to_string(N) + "-" + std::to_string(iter) + ".dat", phi);

        // Energy cant be positiv
        // shift potential

        double shift = 0.0;

        if (E > 0) {
            shift = -20;
            // shift = - 1.2 * E;
            E = energy(phi, V + shift);
            count_shift++;
        }

        Function<double, NDIM> Vphi = (V + shift) * phi;
        Vphi.truncate();

        SeparatedConvolution<double, NDIM> op = BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * E), 0.001, 1e-7);

        Function<double, NDIM> r = phi + 2.0 * op(Vphi); // the residual
        double err = r.norm2();

        // Q = 1 - sum(|phi_prev><phi_prev|) = 1 - |phi_0><phi_0| - |phi_1><phi_1| - ...
        // Q*|Phi> = |Phi> - |phi_0><phi_0|Phi> - |phi_1><phi_1|Phi> - ...

        for (const auto& prev_phi : prev_phi) {
            phi -= inner(prev_phi, phi) * prev_phi;
        }
        phi.scale(1.0 / phi.norm2());

        phi = solver.update(phi, r);

        double norm = phi.norm2();
        phi.scale(1.0 / norm); // phi *= 1.0/norm
        E = energy(phi, V);

        if (madness_process.world->rank() == 0)
            print("iteration", iter, "energy", E, "norm", norm, "error", err);

        if (err < 5e-4)
            break;
    }

    // madness_process.plot("phi-" + std::to_string(N)+ ".dat", phi);

    if (count_shift != 0) {
        std::cout << "Potential was shifted " << count_shift << " times" << std::endl;
    }

    print("Final energy without shift: ", E);
    return phi;
}
template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Eigensolver<NDIM>::get_orbitals(int core_dim, int as_dim, int froz_virt_dim) const {
    std::vector<SavedFct<NDIM>> sav_orbs;
    for (auto i = 0; i < (core_dim + as_dim + froz_virt_dim); ++i) {
        SavedFct<NDIM> sav_orb(orbitals[i]);
        if (i < core_dim) {
            sav_orb.type = "frozen_occ";
        } else if (i < core_dim + as_dim) {
            sav_orb.type = "active";
        } else if (i < core_dim + as_dim + froz_virt_dim) {
            sav_orb.type = "frozen_virt";
        }
        sav_orbs.push_back(sav_orb);
    }
    return sav_orbs;
}

template class Eigensolver<2>;
template class Eigensolver<3>;