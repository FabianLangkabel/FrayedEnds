#include "eigensolver.hpp"

using namespace madness;

Eigensolver3D::Eigensolver3D(double L, long k, double thresh): MadnessProcess(L,k,thresh) {std::cout.precision(6);}

Eigensolver3D::~Eigensolver3D()
{
    V.clear();
    orbitals.clear();
}

// Function to solve the eigenvalue problem for the given potential
void solve(SavedFct input_V, int num_levels, int max_iter) {
    Function<double, 3> V = loadfct(input_V);
    // Create the guess generator
    GuessGenerator<double, 3> guess_generator(world);              // Guess generator for all potentials
    // Create the guess functions
    std::vector<Function<double,3>> guesses = guess_generator.create_guesses(num_levels, V);

    // plot guess functions
    for (int i = 0; i < guesses.size(); i++) {
        plot(guesses[i], "g-" + std::to_string(i) + ".dat");
    }

    // Diagonalize the Hamiltonian matrix
    std::pair<Tensor<double>, std::vector<Function<double, 3>>> tmp = diagonalize(world, guesses, V);
    std::vector<Function<double, 3>> diagonalized_guesses = tmp.second;

    // store the eigenfunctions in vector eigenfunctions
    std::vector<Function<double, 3>> eigenfunctions;

    // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
    for (int i = 0; i < num_levels; i++) {
        Function<double, 3> phi = optimize(world, V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
        eigenfunctions.push_back(phi);
    }

    // diagonalize the eigenfunctions again
    std::pair<Tensor<double>, std::vector<Function<double, 3>>> diagonalized = diagonalize(world, eigenfunctions, V);
    std::cout << "Diagonalize" << std::endl;
    orbitals = diagonalized.second;
}

// Function to solve the eigenvalue problem for the given potential with given guesses
std::vector<Function<double, NDIM>> solve_with_input_guesses(SavedFct input_V, const std::vector<SavedFct>& input_guesses, int num_levels, int max_iter) {
    Function<double, 3> V = loadfct(input_V);
    std::vector<Function<double, 3>> guesses;

    for (const auto& guess : input_guesses) {
        guesses.push_back(loadfct(guess));
    }

    // Diagonalize the Hamiltonian matrix
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(world, guesses, V);
    std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;

    // store the eigenfunctions in vector eigenfunctions
    std::vector<Function<double, NDIM>> eigenfunctions;

    // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
    for (int i = 0; i < num_levels; i++) {
        Function<double, NDIM> phi = optimize(world, V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
        eigenfunctions.push_back(phi);
    }

    // diagonalize the eigenfunctions again
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(world, eigenfunctions, V);
    std::cout << "Diagonalize" << std::endl;
    std::vector<Function<double, NDIM>> y = diagonalized.second;

    return y;
}


// Function to optimize the eigenfunction for each energy level
Function<double, 3> optimize(World& world, Function<double, 3>& V, const Function<double, 3> guess_function, int N, const std::vector<Function<double, 3>>& prev_phi, int max_iter) {

    // Create the initial guess wave function
    Function<double, 3> phi = guess_function;

    phi.scale(1.0/phi.norm2()); // phi *= 1.0/norm
    double E = energy(world, phi, V);

    NonlinearSolverND<3> solver;
    int count_shift = 0; // counter how often the potential was shifted

    for(int iter = 0; iter <= max_iter; iter++) {
        
        // plot("phi-" + std::to_string(N) + "-" + std::to_string(iter) + ".dat", phi);

        // Energy cant be positiv
        // shift potential

        double shift = 0.0;

        if (E > 0) {
            shift = -20;
            //shift = - 1.2 * E;
            E = energy(world, phi, V + shift);
            count_shift++;
        }

        Function<double, 3> Vphi = (V + shift) * phi;
        Vphi.truncate();
        
        SeparatedConvolution<double,3> op = BSHOperator<3>(world, sqrt(-2*E), 0.001, 1e-7);  

        Function<double, 3> r = phi + 2.0 * op(Vphi); // the residual
        T err = r.norm2();

        // Q = 1 - sum(|phi_prev><phi_prev|) = 1 - |phi_0><phi_0| - |phi_1><phi_1| - ...
        // Q*|Phi> = |Phi> - |phi_0><phi_0|Phi> - |phi_1><phi_1|Phi> - ...

        for (const auto& prev_phi : prev_phi) {
            phi -= inner(prev_phi, phi)*prev_phi; 
        }
        phi.scale(1.0/phi.norm2());

        phi = solver.update(phi, r);

        double norm = phi.norm2();
        phi.scale(1.0/norm);  // phi *= 1.0/norm
        E = energy(world,phi,V); 

        if (world.rank() == 0)
            print("iteration", iter, "energy", E, "norm", norm, "error",err);

        if (err < 5e-4) break;
    }

    plot("phi-" + std::to_string(N)+ ".dat", phi);

    if (count_shift != 0) {
        std::cout << "Potential was shifted " << count_shift << " times" << std::endl;
    }

    print("Final energy without shift: ", E);
    return phi;
}