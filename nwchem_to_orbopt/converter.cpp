#include "converter.hpp"

using namespace madness;


Converter::Converter(int argc, char** argv, double L, long k, double thresh)
{
    int arg = 0;
    char **a = new char*[0]();
    initialize(arg, a);
    world = new World(SafeMPI::COMM_WORLD);

    startup(*world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

Converter::~Converter()
{

}

void Converter::create_mos(std::string nwchem_file)
{
    std::ostream bad(nullptr);
    //slymer::NWChem_Interface nwchem(nwchem_file, bad);
    slymer::NWChem_Interface nwchem(nwchem_file, std::cout);

    nwchem.read(slymer::Properties::Atoms | slymer::Properties::Basis | slymer::Properties::Energies | slymer::Properties::MOs | slymer::Properties::Occupancies);

    // Cast the 'basis_set' into a gaussian basis
    // and iterate over it
    std::vector<real_function_3d> aos;
    int i = 0;
    for (auto basis: slymer::cast_basis<slymer::GaussianFunction>(nwchem.basis_set)) {
        // Get the center of gaussian as its special point
        std::vector<coord_3d> centers;
        coord_3d r;
        r[0] = basis.get().center[0];
        r[1] = basis.get().center[1];
        r[2] = basis.get().center[2];
        centers.push_back(r);

        // Now make the function
        aos.push_back(factoryT(*world).functor(functorT(new slymer::Gaussian_Functor(basis.get(), centers))));
    }

    auto molecule = madness::Molecule();
    for (auto atom: nwchem.atoms) {
        molecule.add_atom(atom.position[0], atom.position[1], atom.position[2], (double)symbol_to_atomic_number(atom.symbol), symbol_to_atomic_number(atom.symbol));
    }

    Vnuc = new Nuclear<double,3>(*world, molecule);
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();

    // Transform ao's now
    mos = transform(*world, aos, nwchem.MOs);
    truncate(*world, mos);

}


void Converter::define_as(int number_occupied_orbitals, std::vector<int> active_orbitals)
{
    int occupied_as_orbs = 0;
    // We assume, that the orbitals are sorted energetically!
    for(int i = 0; i < number_occupied_orbitals; i++)
    {
        if(std::find(active_orbitals.begin(), active_orbitals.end(), i) != active_orbitals.end())
        {
            occupied_as_orbs++;
            active_orbs.push_back(mos[i]);
        }
        else
        {
            frozen_occ_orbs.push_back(mos[i]);
        }
    }
    for(int active_orb : active_orbitals)
    {
        if(active_orb >= number_occupied_orbitals){ active_orbs.push_back(mos[active_orb]);}
    }
    as_dim = active_orbs.size();
    core_dim = frozen_occ_orbs.size();
}

void Converter::CalculateAllIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initializing the Coulomb operator
    real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, coulomb_lo, coulomb_eps));

    // Multiplication of AS orbital pairs and their Coulomb element (are needed more often and are therefore stored)
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<real_function_3d> orbs_kl;
    for(int k = 0; k < as_dim; k++)
    {
        std::vector<real_function_3d> kl = active_orbs[k] * active_orbs;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);
    std::vector<real_function_3d> coul_orbs_mn = apply(*world, *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);
    auto t2 = std::chrono::high_resolution_clock::now();

    // AS-AS one electron integrals
    as_integrals_one_body = Eigen::MatrixXd::Zero(as_dim, as_dim);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb_k = D(active_orbs[k]);
                real_function_3d d_orb_l = D(active_orbs[l]);
                as_integrals_one_body(k, l) += 0.5 * inner(d_orb_k,d_orb_l);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(active_orbs[l]);
            as_integrals_one_body(k, l) += inner(active_orbs[k], Vnuc_orb_l);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();


    // AS two electron integrals
    as_integrals_two_body = Eigen::Tensor<double, 4>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*world, orbs_kl, coul_orbs_mn, false);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    as_integrals_two_body(k, m, l, n) = Inner_prods(k*as_dim + l, m*as_dim + n);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();



    // Core-AS two electron integrals <ak|al>
    {
        core_as_integrals_two_body_akal = Eigen::Tensor<double, 3>(core_dim, as_dim, as_dim);
        std::vector<real_function_3d> orbs_aa;
        for (int a = 0; a < core_dim; a++)
        {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        madness::Tensor<double> Inner_prods_akal = matrix_inner(*world, orbs_aa, coul_orbs_mn, false);
        for (int a = 0; a < core_dim; a++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    core_as_integrals_two_body_akal(a, k, l) = Inner_prods_akal(a, k*as_dim + l);
                }
            }
        }
    }

    // Core-AS two electron integrals <ak|la>, <ak|ln>, <ab|ak> and <ba|ak>
    core_as_integrals_two_body_akla = Eigen::Tensor<double, 3>(core_dim, as_dim, as_dim);
    for (int a = 0; a < core_dim; a++) //One core orbital after the other -> Slightly less efficient than all a at the same time, but reduces memory
    {
        std::vector<real_function_3d> orbs_ak = frozen_occ_orbs[a] * active_orbs;
        orbs_ak = truncate(orbs_ak, truncation_tol);
        std::vector<real_function_3d> coul_orbs_ak = apply(*world, *coul_op_parallel, orbs_ak);
        coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

        std::vector<real_function_3d> orbs_ka = active_orbs * frozen_occ_orbs[a];
        orbs_ka = truncate(orbs_ka, truncation_tol);
        
        // <ak|la> = <ka|al>
        madness::Tensor<double> Inner_prods_akla = matrix_inner(*world, orbs_ka, coul_orbs_ak, false);
        for(int k = 0; k < as_dim; k++)
        {
            for(int l = 0; l < as_dim; l++)
            {
                core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();


    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Integral timings:" << std::endl;
    std::cout << "Preparation AS pairs: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "AS-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " seconds" << std::endl;
    std::cout << "AS-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " seconds" << std::endl;
    std::cout << "Core-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() << " seconds" << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << " seconds" << std::endl;
}

void Converter::CalculateCoreEnergy()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    double nocc = 2; //Spatial orbitals = 2, Spin orbitals = 1

    //1e Part
    double core_kinetic_energy = 0;
    double core_nuclear_attraction_energy = 0;
    for(int i = 0; i < frozen_occ_orbs.size(); i++)
    {
        //Kinetic
        for (int axis=0; axis<3; axis++) {
            real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
            real_function_3d d_orb = D(frozen_occ_orbs[i]);
            core_kinetic_energy += 0.5 * inner(d_orb, d_orb);
        }
        //Nuclear
        real_function_3d Vnuc_orb = (*Vnuc)(frozen_occ_orbs[i]);
        core_nuclear_attraction_energy += inner(frozen_occ_orbs[i], Vnuc_orb);
    }
    core_kinetic_energy = nocc * core_kinetic_energy;
    core_nuclear_attraction_energy = nocc * core_nuclear_attraction_energy;

    //2e Part
    double core_two_electron_energy = 0;

    real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, coulomb_lo, coulomb_eps));

    // <ab|ab>
    {
        std::vector<real_function_3d> orbs_aa;
        for (int a = 0; a < core_dim; a++)
        {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        std::vector<real_function_3d> coul_orbs_aa = apply(*world, *coul_op_parallel, orbs_aa);
        coul_orbs_aa = truncate(coul_orbs_aa, truncation_tol);
        for (int a = 0; a < core_dim; a++)
        {
            madness::Tensor<double> Inner_prods_abab = matrix_inner(*world, std::vector<real_function_3d>{ orbs_aa[a] }, coul_orbs_aa, false);
            for (int b = 0; b < core_dim; b++)
            {
                core_two_electron_energy += 2 * Inner_prods_abab(0, b);
            }
        }
    }

    for (int a = 0; a < core_dim; a++) //One core orbital after the other -> Slightly less efficient than all a at the same time, but reduces memory
    {
        std::vector<real_function_3d> orbs_ab = frozen_occ_orbs[a] * frozen_occ_orbs;
        orbs_ab = truncate(orbs_ab, truncation_tol);
        std::vector<real_function_3d> coul_orbs_ab = apply(*world, *coul_op_parallel, orbs_ab);
        coul_orbs_ab = truncate(coul_orbs_ab, truncation_tol);
        for (int b = 0; b < core_dim; b++)
        {
            core_two_electron_energy -= inner(orbs_ab[b], coul_orbs_ab[b]);
        }
    }

    core_two_electron_energy = 0.5 * nocc * core_two_electron_energy;

    core_total_energy = core_kinetic_energy + core_nuclear_attraction_energy + core_two_electron_energy;
    print("                   Core - Kinetic energy ", core_kinetic_energy);
    print("        Core - Nuclear attraction energy ", core_nuclear_attraction_energy);
    print("              Core - Two-electron energy ", core_two_electron_energy);
    print("                       Total core energy ", core_total_energy);
    print("                Nuclear repulsion energy ", nuclear_repulsion_energy);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateCoreEnergy took " << duration.count() << " seconds" << std::endl;
}



void Converter::SaveEffectiveHamiltonian(std::string OutputPath)
{
    std::vector<double> effective_one_body_integrals_elements;
    std::vector<double> effective_two_body_integrals_elements;

    Eigen::MatrixXd effective_one_body_integrals = as_integrals_one_body;
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int a = 0; a < core_dim; a++)
            {
                effective_one_body_integrals(k, l) += 0.5 * nocc * (2 * core_as_integrals_two_body_akal(a,k,l) - core_as_integrals_two_body_akla(a,k,l));
            }
            effective_one_body_integrals_elements.push_back(effective_one_body_integrals(k,l));
        }
    }
    
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    effective_two_body_integrals_elements.push_back(as_integrals_two_body(k,l,m,n));
                }
            }
        }
    }

    std::ofstream c_file;
    c_file.open(OutputPath + "/c.txt");
    c_file << std::setprecision (15) << (core_total_energy + nuclear_repulsion_energy);
    c_file.close();

    std::vector<unsigned long> one_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim};
    const npy::npy_data<double> one_e_data{effective_one_body_integrals_elements, one_e_ints_shape, false};
    npy::write_npy(OutputPath + "/htensor.npy", one_e_data);

    std::vector<unsigned long> two_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim, (unsigned long)as_dim, (unsigned long)as_dim};
    const npy::npy_data<double> two_e_data{effective_two_body_integrals_elements, two_e_ints_shape, false};
    npy::write_npy(OutputPath + "/gtensor.npy", two_e_data);
}

void Converter::save_orbitals(std::string output_folder)
{
    int orb_idx = 0;
    for(int i = 0; i < core_dim; i++)
    {
        std::string filename = "orbital_" + std::to_string(i);
        save(frozen_occ_orbs[i], output_folder + "/" + filename); // ohne das .00000 im filename
        orb_idx++;
    }
    for(int i = 0; i < as_dim; i++)
    {
        std::string filename = "orbital_" + std::to_string(i);
        save(active_orbs[i], output_folder + "/" + filename); // ohne das .00000 im filename
        orb_idx++;
    }
}