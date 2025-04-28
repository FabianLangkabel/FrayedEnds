#include "spinorb_optimizer.hpp"

using namespace madness;

SpinorbOpt::SpinorbOpt(int argc, char** argv, double L, long k, double thresh)
{
    int arg = 0;
    char **a = new char*[0]();
    initialize(arg, a);
    world = new World(SafeMPI::COMM_WORLD);
    this->atoms = atoms;

    startup(*world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

SpinorbOpt::~SpinorbOpt()
{

}

void SpinorbOpt::CreateNuclearPotentialAndRepulsion(std::string GeometryFile)
{
    auto molecule = madness::Molecule();
    molecule.read_file(GeometryFile);  //Define Molecule
    Vnuc = new Nuclear<double,3>(*world, molecule);
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();
    std::cout << "Enuc: " << endl << nuclear_repulsion_energy << endl;
}

void SpinorbOpt::ReadInitialOrbitals(std::vector<Orbital> alpha_orbs, std::vector<Orbital> beta_orbs, int num_alpha_orbs, int num_beta_orbs)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<Orbital> all_alpha_orbitals = alpha_orbs;
    active_alpha_orbital_indicies.resize(num_alpha_orbs);
    std::fill(active_alpha_orbital_indicies.begin(), active_alpha_orbital_indicies.end(), -1);
    frozen_occupied_alpha_orbital_indicies.clear();

    std::vector<Orbital> all_beta_orbitals = beta_orbs;
    active_beta_orbital_indicies.resize(num_beta_orbs);
    std::fill(active_beta_orbital_indicies.begin(), active_beta_orbital_indicies.end(), -1);
    frozen_occupied_beta_orbital_indicies.clear();

    int max_dim = std::max(num_alpha_orbs, num_beta_orbs);
    
    for(int i = 0; i < max_dim; ++i) {
        real_function_3d orb = real_factory_3d(*world);
        if (i < num_alpha_orbs) {
            load(orb, all_alpha_orbitals[i].orbital_file_name);
            all_alpha_orbitals[i].function = orb;
            if(all_alpha_orbitals[i].orbital_type == "active") {
                active_alpha_orbital_indicies[all_alpha_orbitals[i].active_space_index] = i;
                active_alpha_beta_orbs.push_back(all_alpha_orbitals[i].function);
            }
            else if(all_alpha_orbitals[i].orbital_type == "frozen_occupied") {
                frozen_occupied_alpha_orbital_indicies.push_back(i);
            }
        }
        if (i < num_beta_orbs) {
            load(orb, all_beta_orbitals[i].orbital_file_name);
            all_beta_orbitals[i].function = orb;
            if(all_beta_orbitals[i].orbital_type == "active") {
                active_beta_orbital_indicies[all_beta_orbitals[i].active_space_index] = i;
                active_alpha_beta_orbs.push_back(all_beta_orbitals[i].function);
                }
            else if(all_beta_orbitals[i].orbital_type == "frozen_occupied") {
            frozen_occupied_beta_orbital_indicies.push_back(i);
            }
        }
    }

    //Print frozen occupied indicies
    std::cout << "Frozen occupied alpha orbitals: ";
    for (auto const& FroIdxa : frozen_occupied_alpha_orbital_indicies)
    {
        std::cout << FroIdxa << " ";
    }
    std::cout << std::endl;
    std::cout << "Frozen occupied beta orbitals: ";
    for (auto const& FroIdxb : frozen_occupied_beta_orbital_indicies)
    {
        std::cout << FroIdxb << " ";
    }
    std::cout << std::endl;

    //Print active space indicies
    std::cout << "Active space alpha indicies: ";
    for (auto const& AIdxa : active_alpha_orbital_indicies)
    {
        std::cout << AIdxa << " ";
    }
    std::cout << std::endl;
    std::cout << "Active space beta indicies: ";
    for (auto const& AIdxb : active_beta_orbital_indicies)
    {
        std::cout << AIdxb << " ";
    }
    std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadOrbitals took " << duration.count() << " seconds" << std::endl; 
    //std::cout << "Alpha Beta orbs " << active_alpha_beta_orbs.size() << std::endl;
    
} 

void SpinorbOpt::ReadRDMFiles(std::vector<std::string> one_body_rdm_files, std::vector<std::string> two_body_rdm_files)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    //read the numpy files
    auto alpha_rdm_npy = npy::read_npy<double>(one_body_rdm_files[0]);
    auto beta_rdm_npy = npy::read_npy<double>(one_body_rdm_files[1]);
    auto alpha_alpha_rdm_npy = npy::read_npy<double>(two_body_rdm_files[0]);
    auto alpha_beta_rdm_npy = npy::read_npy<double>(two_body_rdm_files[1]);
    auto beta_beta_rdm_npy = npy::read_npy<double>(two_body_rdm_files[2]);

    std::vector<double> data_alpha = alpha_rdm_npy.data; //extract the data and recast as std::vector
    std::vector<double> data_beta = beta_rdm_npy.data;
    std::vector<double> data_alpha_alpha = alpha_alpha_rdm_npy.data;
    std::vector<double> data_alpha_beta = alpha_beta_rdm_npy.data;
    std::vector<double> data_beta_beta = beta_beta_rdm_npy.data;
    
    //int dim_alpha = all_alpha_orbitals.size();
    //int dim_beta = all_beta_orbitals.size();
    int max_dim = active_alpha_beta_orbs.size();

    //Transform the std::vector to Eigen::Matrix
    Eigen::VectorXd Alpha_Rdm_Vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data_alpha.data(), data_alpha.size());
    Eigen::VectorXd Beta_Rdm_Vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data_beta.data(), data_beta.size());
    Eigen::VectorXd Alpha_Alpha_Rdm_Vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data_alpha_alpha.data(), data_alpha_alpha.size());
    Eigen::VectorXd Alpha_Beta_Rdm_Vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data_alpha_beta.data(), data_alpha_beta.size());
    Eigen::VectorXd Beta_Beta_Rdm_Vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data_beta_beta.data(), data_beta_beta.size());


    //Insert the rdm vectors into the blownup matrix and tensor, both are in phys notation
    
    Alpha_Beta_Rdm_Matrix = Eigen::MatrixXd::Zero(max_dim, max_dim);
    int x = 0;
    int y = 0;
    for (int i = 0; i < max_dim; i+=2) {
        for (int j = 0; j < max_dim; j+=2) {
            Alpha_Beta_Rdm_Matrix(i, j) = Alpha_Rdm_Vector(x++);
        }
    }
    for (int i = 1; i < max_dim; i+=2) {
        for (int j = 1; j < max_dim; j+=2) {
            Alpha_Beta_Rdm_Matrix(i, j) = Beta_Rdm_Vector(y++);
        }
    }

    Alpha_Beta_Rdm_Tensor = Eigen::Tensor<double, 4>(max_dim, max_dim, max_dim, max_dim);
    Alpha_Beta_Rdm_Tensor.setZero();
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    for (int i = 0; i < max_dim; i+=2) {
        for (int j = 0; j < max_dim; j+=2) {
            for (int k = 0; k < max_dim; k+=2) {
                for (int l = 0; l < max_dim; l+=2) {
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = Alpha_Alpha_Rdm_Vector(a++);
                }
            }    
        }
    }
    for (int i = 0; i < max_dim; i+=2) {
        for (int j = 1; j < max_dim; j+=2) {
            for (int k = 1; k < max_dim; k+=2) {
                for (int l = 0; l < max_dim; l+=2) {
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = Alpha_Beta_Rdm_Vector(b++);
                    Alpha_Beta_Rdm_Tensor(j, i, k, l) = Alpha_Beta_Rdm_Vector(c++);
                }
            }    
        }
    }
    for (int i = 1; i < max_dim; i+=2) {
        for (int j = 1; j < max_dim; j+=2) {
            for (int k = 1; k < max_dim; k+=2) {
                for (int l = 1; l < max_dim; l+=2) {
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = Beta_Beta_Rdm_Vector(d++);
                }
            }    
        }
    }

   
    //reshape the Tensor into chemical notation

    //std::cout << "Alpha Beta Rdms Matrix: " << endl << Alpha_Beta_Rdm_Matrix << endl;
    //std::cout << "Alpha Beta combined Rdms size: " << endl << Alpha_Beta_Rdm_Tensor.size() << endl;

    //int nonzero = 0;
    /*for (int i = 0; i < max_dim; ++i) {
        for (int j = 0; j < max_dim; ++j) {
            for (int k = 0; k < max_dim; ++k) {
                for (int l = 0; l < max_dim; ++l) {
                    if (Alpha_Beta_Rdm_Tensor(i, j, k, l) != 0) {
                        std::cout << "Alpha_Beta_rdms[" << i << "][" << j << "][" << k << "][" << l << "] = " 
                                  << Alpha_Beta_Rdm_Tensor(i, j, k, l) << std::endl;
                    }
                }
            }
        }
    }*/

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadRDMFiles took " << duration.count() << " seconds" << std::endl; 
}


/*void SpinorbOpt::TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix)
{
    *ObjectMatrix = TransformationMatrix.transpose() * *ObjectMatrix * TransformationMatrix;
}

void SpinorbOpt::TransformTensor(Eigen::Tensor<double, 4>* ObjectTensor, Eigen::MatrixXd TransformationMatrix)
{
    int n = TransformationMatrix.rows();
    Eigen::Tensor<double, 4> temp1(n, n, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < n; l++) {
                    double k_value = 0;
                    for (int k = 0; k < n; k++) {
                        k_value += TransformationMatrix(k, k2) * ObjectTensor->coeff(i, j, k, l);
                    }
                    temp1(i, j, k2, l) = k_value;
                }
            }
        }
    }

    Eigen::Tensor<double, 4> temp2(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < n; l++) {
                    double i_value = 0;
                    for (int i = 0; i < n; i++) {
                        i_value += TransformationMatrix(i, i2) * temp1(i, j, k2, l);
                    }
                    temp2(i2, j, k2, l) = i_value;
                }
            }
        }
    }

    Eigen::Tensor<double, 4> temp3(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < n; l2++) {
                    double l_value = 0;
                    for (int l = 0; l < n; l++) {
                        l_value += TransformationMatrix(l, l2) * temp2(i2, j, k2, l);
                    }
                    temp3(i2, j, k2, l2) = l_value;
                }
            }
        }
    }

    Eigen::Tensor<double, 4> temp4(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j2 = 0; j2 < n; j2++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < n; l2++) {
                    double j_value = 0;
                    for (int j = 0; j < n; j++) {
                        j_value += TransformationMatrix(j, j2) * temp3(i2, j, k2, l2);
                    }
                    temp4(i2, j2, k2, l2) = j_value;
                }
            }
        }
    }
    *ObjectTensor = temp4;
}*/

/void SpinorbOpt::TransformToNObasis()
{
    int num_active_orbs = active_alpha_beta_orbs.size();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(Alpha_Beta_Rdm_Matrix);
    ActiveSpaceRotationMatrix = es.eigenvectors().rowwise().reverse();
    TransformMatrix(&Alpha_Beta_Rdm_Matrix, ActiveSpaceRotationMatrix);
    TransformTensor(&Alpha_Beta_Rdm_Tensor, ActiveSpaceRotationMatrix);
    madness::Tensor<double> T(num_active_orbs, num_active_orbs);
    for (int i = 0; i < num_active_orbs; i++) {
        for (int j = 0; j < num_active_orbs; j++) {
            T(i,j) = ActiveSpaceRotationMatrix(i,j);
        }
    }
    
    
    std::cout << "dim alpha beta " << num_active_orbs << std::endl;
    
    std::vector<real_function_3d> orbitals_rotate; 
    for (int i = 0; i < num_active_orbs; i++)
        {
            orbitals_rotate.push_back(active_alpha_beta_orbs[i]);
        }
    
    orbitals_rotate = transform(*world, orbitals_rotate, T);

    for (int i = 0; i < num_active_orbs; i++)
    {
        active_alpha_beta_orbs[i] = orbitals_rotate[i];
        //orbitals_rotate.push_back(all_beta_orbitals[actIdxA].function);
    }
    
    std::cout << "orbitals_rotate " << orbitals_rotate.size() << std::endl;
    std::cout << "alpha beta orbs " << active_alpha_beta_orbs.size() << std::endl;
    
}

void SpinorbOpt::CalculateAllIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);

    int dim_ab = active_alpha_beta_orbs.size();

    //std::cout << "Alpha Beta orbs " << alpha_beta_orbitals.size() << std::endl;

    //Set up the one-body integrals matrix and the two body integrals tensor

    integrals_kinetic.clear();
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim_ab, dim_ab)); //x
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim_ab, dim_ab)); //y
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim_ab, dim_ab)); //z
    integrals_potential = Eigen::MatrixXd::Zero(dim_ab, dim_ab);
    integrals_two_body = Eigen::Tensor<double, 4>(dim_ab, dim_ab, dim_ab, dim_ab);
    integrals_two_body.setZero();


    //Calculate the one-body integrals

    for(int k = 0; k < dim_ab; k++)
    {
        //std::cout << k << std::endl;
        //Kinetic
        for (int axis=0; axis<3; axis++) {
            real_derivative_3d D_ab = free_space_derivative<double,3>(*world, axis);
            real_function_3d dab_orb_left = D_ab(active_alpha_beta_orbs[k]);
            real_function_3d dab_orb_right = D_ab(active_alpha_beta_orbs[k]);
            integrals_kinetic[axis](k, k) = 0.5 * inner(dab_orb_left,dab_orb_right);
        }
        //Nuclear
        real_function_3d Vnuc_orb_l = (*Vnuc)(active_alpha_beta_orbs[k]);
        integrals_potential(k, k) = inner(active_alpha_beta_orbs[k], Vnuc_orb_l);
    }

    integrals_one_body = integrals_potential + integrals_kinetic[0] + integrals_kinetic[1] + integrals_kinetic[2];

    //Construct (kl| orbitals; be aware of the indices, not all elements are meaningful and will be neglected

    ab_orbs_kl.clear();
    for(int k = 0; k < dim_ab; ++k)
    {
        std::vector<real_function_3d> kl = active_alpha_beta_orbs[k] * active_alpha_beta_orbs;
        ab_orbs_kl.insert(std::end(ab_orbs_kl), std::begin(kl), std::end(kl));
    }
    ab_orbs_kl = truncate(ab_orbs_kl, 1e-5);

    //construct 1/r|mn) orbitals

    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, 0.001, 1e-6));
    ab_coul_orbs_mn = apply(*world, *coul_op_parallel, ab_orbs_kl);
    ab_coul_orbs_mn = truncate(ab_coul_orbs_mn, 1e-5);

    //calculate and insert the non-zero elements to the two-body integral tensor, in phys notation already

    madness::Tensor<double> Inner_prods = matrix_inner(*world, ab_orbs_kl, ab_coul_orbs_mn, false);

    //alpha alpha elements
    for(int k = 0; k < dim_ab; k+=2)
    {
        for(int l = 0; l < dim_ab; l+=2)
        {
            if (k == l) {
                integrals_two_body(k, k, k, k) = Inner_prods(k*dim_ab + k, k*dim_ab + k);
            }
            if (k != l) {
                integrals_two_body(k, l, l, k) = Inner_prods(k*dim_ab + l, l*dim_ab + k);
                integrals_two_body(k, l, k, l) = Inner_prods(k*dim_ab + k, l*dim_ab + l);
                integrals_two_body(k, k, l, l) = Inner_prods(k*dim_ab + l, k*dim_ab + l);
            }    
        }      
    }

    //beta beta elements
    for(int k = 1; k < dim_ab; k+=2)
    {
        for(int l = 1; l < dim_ab; l+=2)
        {
            if (k == l) {
                integrals_two_body(k, k, k, k) = Inner_prods(k*dim_ab + k, k*dim_ab + k);
            }
            if (k != l) {
                integrals_two_body(k, k, l, l) = Inner_prods(k*dim_ab + l, k*dim_ab + l);
                integrals_two_body(k, l, l, k) = Inner_prods(k*dim_ab + l, l*dim_ab + k);
                integrals_two_body(k, l, k, l) = Inner_prods(k*dim_ab + k, k*dim_ab + k);
            }    
        }
    }

    //alpha beta elements
    for(int k = 0; k < dim_ab; k+=2)
    {
        for(int l = 1; l < dim_ab; l+=2)
        {
            integrals_two_body(k, l, k, l) = Inner_prods(k*dim_ab + k, l*dim_ab + l);
            integrals_two_body(l, k, l, k) = Inner_prods(l*dim_ab + l, k*dim_ab + k);
        }
    }
    
    for(int k = 0; k < dim_ab; k+=2)
    {
        for(int l = 0; l < dim_ab; l+=2)
        {
            for(int m = 1; m < dim_ab; m+=2)
            {
                for(int n = 1; n < dim_ab; n+=2)
                {
                    if (k != l && m != n) {
                        integrals_two_body(k, m, l, n) = Inner_prods(k*dim_ab + l, m*dim_ab + n);
                        integrals_two_body(m, k, n, l) = Inner_prods(m*dim_ab + n, k*dim_ab + l);
                    }  
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateAllIntegrals took " << duration.count() << " seconds" << std::endl;

    //std::cout << "Two body integrals size " << integrals_two_body.size() << std::endl;
 
    /*int nonzero = 0;
    for (int i = 0; i < dim_ab; i+=2) {
        for (int j = 0; j < dim_ab; j+=2) {
            for (int k = 0; k < dim_ab; k+=2) {
                for (int l = 0; l < dim_ab; l+=2) {
                    if (integrals_two_body(i, j, k, l) != 0) {
                        nonzero +=1;
                        std::cout << "Alpha_Beta_integrals[" << i << "][" << j << "][" << k << "][" << l << "] = " 
                                  << integrals_two_body(i, j, k, l) << std::endl;
                    }
                }
            }
        }
    }
    //std::cout << "Nonzero elements: " << nonzero << std::endl;*/
}


void SpinorbOpt::CalculateEnergy()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int dim_ab = Alpha_Beta_Rdm_Matrix.rows();
    std::cout << "dim ab calc en" << dim_ab << std::endl;
    double kinetic_energy = 0.0;
    double nuclear_attraction_energy = 0.0;

    //One Particle Part
    for(int k = 0; k < dim_ab; k++)
    {
        for(int l = 0; l < dim_ab; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                kinetic_energy += Alpha_Beta_Rdm_Matrix(k, l) * integrals_kinetic[axis](k, l);
            }
            //Nuclear
            nuclear_attraction_energy += Alpha_Beta_Rdm_Matrix(k, l) * integrals_potential(k, l);
        }
    }
    
    //Two Particle Part
    double two_electron_energy = 0.0;    
    for(int i = 0; i < dim_ab; i++) {
        for(int j = 0; j < dim_ab; j++) {
            for(int k = 0; k < dim_ab; k++) {
                for(int l = 0; l < dim_ab; l++){
                    two_electron_energy += Alpha_Beta_Rdm_Tensor(i, j, k, l) * 0.5 * integrals_two_body(i, j, k, l); 
                }
            }
        }
    }

    double total_energy = kinetic_energy + two_electron_energy + nuclear_attraction_energy + nuclear_repulsion_energy;
    
    print("            Kinetic energy ", kinetic_energy);
    print(" Nuclear attraction energy ", nuclear_attraction_energy);
    print("       Two-electron energy ", two_electron_energy);
    print(" Nuclear  repulsion energy ", nuclear_repulsion_energy);
    print("              Total energy ", total_energy);


    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateEnergies took " << duration.count() << " seconds" << std::endl;
}

/*void SpinorbOpt::CalculateLagrangeMultiplier()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int dim = all_alpha_orbitals.size() + all_beta_orbitals.size();
    LagrangeMultiplier = Eigen::MatrixXd::Zero(dim, dim);
    for(int a = 0; a < dim; a++)
    {
        for(int i = 0; i < dim; i++)
        {
            LagrangeMultiplier(a, i) = CalculateLagrangeMultiplierElement(dim, a, i);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
}*/

/*double SpinorbOpt::CalculateLagrangeMultiplierElement(int dim, int a, int i)
{
    double element = integrals_potential(a, i);
    for (int axis=0; axis<3; axis++) {
        element += integrals_kinetic[axis](a, i);
    }
    element = full_one_rdm(i,i) * element;

    for(int l = 0; l < dim; l++)
    {
        for(int n = 0; n < dim; n++)
        {
            for(int k = 0; k < dim; k++)
            {
                element += full_two_rdm(k, l, i, n) * integrals_two_body(a, k, l, n);
            }
        }
    }
    return element;
}*/

/*void SpinorbOpt::OptimizeSpinorbitals(double optimization_thresh, double NO_occupation_thresh)
{
    int number_active_orbitals = active_orbital_indicies.size();
    std::vector<real_function_3d> frozen_orbs;
    for (auto const& frozenIdx : frozen_occupied_orbital_indicies)
    {
        frozen_orbs.push_back(all_orbitals[frozenIdx].function);
    }
    std::vector<real_function_3d> orbitals;
    for (auto const& orb : all_orbitals)
    {
        orbitals.push_back(orb.function);
    }

    bool converged = false;
    int iterstep = 0;
    while(!converged)
    {
        iterstep++;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;

        //Update LagrangeMultiplier
        CalculateLagrangeMultiplier();

        //Update orbitals
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_error = 0;

        //Solver update
        //std::vector<real_function_3d> rs;
        //for(int i = number_frozen_orbs; i < number_orbitals; i++) {
        //    rs.push_back(GetOrbitalUpdate(i));
        //}
        //orbitals = solver->update(orbitals, rs);

        //Fixed-Point update new
        std::vector<int> orbital_indicies_for_update;
        for (int idx = 0; idx < active_orbital_indicies.size(); idx++)
        {
            int actIdx = active_orbital_indicies[idx];
            if(abs(full_one_rdm(actIdx, actIdx)) >= NO_occupation_thresh)
            {
                orbital_indicies_for_update.push_back(actIdx);
            }
            else
            {
                std::cout << "Skip optimization of orbital " << actIdx << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")" << std::endl;
            }
        }

        std::vector<real_function_3d> AllActiveOrbitalUpdates = GetAllActiveOrbitalUpdates(orbital_indicies_for_update);
        for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
        {
            int actIdx = orbital_indicies_for_update[idx];
            orbitals[actIdx] = orbitals[actIdx] - AllActiveOrbitalUpdates[idx];
        }

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        std::cout << "Orthonormalize orbitals" << std::endl;
        //Project out FrozenOrbitals
        if(frozen_orbs.size() > 0)
        {
            auto Q = QProjector(*world, frozen_orbs);
            for (auto const& actIdx : active_orbital_indicies)
            {
                orbitals[actIdx] = Q(orbitals[actIdx]);
            }
        }

        //Orthonormalize orbitals
        orbitals = orthonormalize_symmetric(orbitals);
        //orbitals = orthonormalize_cd(orbitals);
        for(int i = 0; i < orbitals.size(); i++)
        {
            orbitals[i] = orbitals[i].truncate(1e-5);
        }

        //Update all_orbitals
        for (int i = 0; i < all_orbitals.size(); i++)
        {
            all_orbitals[i].function = orbitals[i];
        }

        std::cout << "Update Integrals" << std::endl;
        //Update integrals for new orbitals
        UpdateIntegrals();
        //CalculateAllIntegrals();

        //Calculate new energy
        CalculateEnergies();

        //Check convergence
        std::cout << "Highest error: " << highest_error << std::endl;
        if(highest_error < optimization_thresh){converged = true;}
    }
}*/

/*std::vector<real_function_3d> SpinorbOpt::GetAllActiveSpinorbitalUpdates(std::vector<int> orbital_indicies_for_update)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    int dim = all_orbitals.size();

    std::vector<real_function_3d> AllOrbitalUpdates;
    
    //Calculate rdm_ii_inv values
    std::vector<double> rdm_ii_inv;
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];
        rdm_ii_inv.push_back(1 / full_one_rdm(i,i));
    }

    //1e Part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];
        real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);
        real_function_3d rhs = (*Vnuc)(all_orbitals[i].function);
        for(int k = 0; k < dim; k++)
        {
            if(k != i)
            {
                rhs -= rdm_ii_inv[idx] * LagrangeMultiplier(k,i) * all_orbitals[k].function;
            }
        }
        AllOrbitalUpdates.push_back(rhs);
    }
    AllOrbitalUpdates = truncate(AllOrbitalUpdates, 1e-5);

    //2e Part
    //auto t2 = std::chrono::high_resolution_clock::now();
    for(int k = 0; k < dim; k++)
    {
        std::vector<real_function_3d> lnk = coul_orbs_mn * all_orbitals[k].function;
        lnk = truncate(lnk, 1e-5);
        for(int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
        {
            int i = orbital_indicies_for_update[idx];
            std::vector<real_function_3d> lnk_copy = copy(*world, lnk, false);
            for(int l = 0; l < dim; l++)
            {
                for(int n = 0; n < dim; n++)
                {
                    lnk_copy[l*dim + n] *= full_two_rdm(k, l, i, n) * rdm_ii_inv[idx];
                }
            }
            AllOrbitalUpdates[idx] += sum(*world, lnk_copy);
        }
    }
    //auto t3 = std::chrono::high_resolution_clock::now();

    //BSH part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];
        double en = LagrangeMultiplier(i,i) * rdm_ii_inv[idx];
        SeparatedConvolution<double,3> bsh_op = BSHOperator<3>(*world, sqrt(-2*en), 0.01, 1e-6);
        real_function_3d r = all_orbitals[i].function + 2.0 * bsh_op(AllOrbitalUpdates[idx]); // the residual
        double err = r.norm2();
        std::cout << "Error of Orbital " << i << ": " << err << std::endl; 
        if(err > highest_error){highest_error = err; }
        AllOrbitalUpdates[idx] = r;
    }
    

    auto t4 = std::chrono::high_resolution_clock::now();
    auto t41 = std::chrono::duration_cast<std::chrono::seconds>(t4 - t1);
    //auto t32 = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2);
    std::cout << "GetAllActiveSpinorbitalUpdates took " << t41.count() << " seconds" << std::endl;
    //std::cout << "t32 took " << t32.count() << " seconds" << std::endl;

    return AllSpinorbitalUpdates;
} */


 /*void SpinorbOpt::SaveNOs(std::string OutputPath)
{
    for(int i = 0; i < all_alpha_orbitals.size(); i++)
    {
        save(all_alpha_orbitals[i].function, OutputPath + "/alpha_NO_" + std::to_string(i)); // ohne das .00000 im filename
    }
    
    for(int i = 0; i < all_beta_orbitals.size(); i++)
    {
        save(all_beta_orbitals[i].function, OutputPath + "/beta_NO_" + std::to_string(i)); // ohne das .00000 im filename
    }
}

void SpinorbOpt::SaveSpinorbitals(std::string OutputPath)
{
    for(int i = 0; i < all_alpha_orbitals.size(); i++)
    {
        std::string base_filename = all_alpha_orbitals[i].orbital_file_name.substr(all_alpha_orbitals[i].orbital_file_name.find_last_of("/\\") + 1);
        save(all_alpha_orbitals[i].function, OutputPath + "/alpha_" + base_filename); // ohne das .00000 im filename
    }
    
    for(int i = 0; i < all_beta_orbitals.size(); i++)
    {
        std::string base_filename = all_beta_orbitals[i].orbital_file_name.substr(all_beta_orbitals[i].orbital_file_name.find_last_of("/\\") + 1);
        save(all_beta_orbitals[i].function, OutputPath + "/beta_" + base_filename); // ohne das .00000 im filename
    }
}

void SpinorbOpt::SaveIntegralsToNumpy(std::string OutputPath)
{
    int num_alpha_orbs = all_alpha_orbitals.size();
    int num_beta_orbs = all_beta_orbitals.size();
    int dim_ab = num_alpha_orbs + num_beta_orbs;

    std::vector<double> one_alpha_e_int_elements;
    for(int i = 0; i < dim_ab; i+=2)
    {
        for(int j = 0; j < dim_ab; j+=2)
        {
            one_alpha_e_int_elements.push_back(integrals_one_body(i,j));
        }
    }
    std::vector<double> one_beta_e_int_elements;
    for(int a = 1; a < dim_ab; a+=2)
    {
        for(int b = 1; b < dim_ab; b+=2)
        {
            one_beta_e_int_elements.push_back(integrals_one_body(a,b));
        }
    }

    std::vector<double> two_e_integrals_aa_elements;
    for(int a = 0; a < dim_ab; a+=2)
    {
        for(int b = 0; b < dim_ab; b+=2)
        {
            for(int c = 0; c < dim_ab; c+=2)
            {
                for(int d = 0; d < dim_ab; d+=2)
                {
                    two_e_integrals_aa_elements.push_back(integrals_two_body(a,c,b,d));
                }
            }
        }
    }

    std::vector<double> two_e_integrals_bb_elements;
    for(int a = 1; a < dim_ab; a+=2)
    {
        for(int b = 1; b < dim_ab; b+=2)
        {
            for(int c = 1; c < dim_ab; c+=2)
            {
                for(int d = 1; d < dim_ab; d+=2)
                {
                    two_e_integrals_bb_elements.push_back(integrals_two_body(a,c,b,d));
                }
            }
        }
    }

    std::vector<double> two_e_integrals_ab_elements;
    for(int a = 0; a < dim_ab; a+=2)
    {
        for(int b = 0; b < dim_ab; b+=2)
        {
            for(int c = 1; c < dim_ab; c+=2)
            {
                for(int d = 1; d < dim_ab; d+=2)
                {
                    two_e_integrals_ab_elements.push_back(integrals_two_body(a,c,b,d));
                }
            }
        }
    }


    std::vector<unsigned long> one_alpha_e_ints_shape{(unsigned long)num_alpha_orbs, (unsigned long)num_alpha_orbs};
    const npy::npy_data<double> one_alpha_e_data{one_alpha_e_int_elements, one_alpha_e_ints_shape, false};
    npy::write_npy(OutputPath + "/alpha_htensor.npy", one_alpha_e_data);

    std::vector<unsigned long> one_beta_e_ints_shape{(unsigned long)num_beta_orbs, (unsigned long)num_beta_orbs};
    const npy::npy_data<double> one_beta_e_data{one_beta_e_int_elements, one_beta_e_ints_shape, false};
    npy::write_npy(OutputPath + "/beta_htensor.npy", one_beta_e_data);

    std::vector<unsigned long> two_e_integrals_aa_elements_shape{(unsigned long)num_alpha_orbs, (unsigned long)num_alpha_orbs, (unsigned long)num_alpha_orbs, (unsigned long)num_alpha_orbs};
    const npy::npy_data<double> two_alpha_alpha_e_data{two_e_integrals_aa_elements, two_e_integrals_aa_elements_shape, false};
    npy::write_npy(OutputPath + "/alpha_alpha_gtensor.npy", two_alpha_alpha_e_data);

    std::vector<unsigned long> two_e_integrals_bb_elements_shape{(unsigned long)num_beta_orbs, (unsigned long)num_beta_orbs, (unsigned long)num_beta_orbs, (unsigned long)num_beta_orbs};
    const npy::npy_data<double> two_beta_beta_e_data{two_e_integrals_bb_elements, two_e_integrals_bb_elements_shape, false};
    npy::write_npy(OutputPath + "/beta_beta_gtensor.npy", two_beta_beta_e_data);

    std::vector<unsigned long> two_e_integrals_ab_elements_shape{(unsigned long)num_alpha_orbs, (unsigned long)num_alpha_orbs, (unsigned long)num_beta_orbs, (unsigned long)num_beta_orbs};
    const npy::npy_data<double> two_alpha_beta_e_data{two_e_integrals_ab_elements, two_e_integrals_ab_elements_shape, false};
    npy::write_npy(OutputPath + "/alpha_beta_gtensor.npy", two_alpha_beta_e_data);

}*/
