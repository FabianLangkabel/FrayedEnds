#include "spinorb_optimizer.hpp"
using json = nlohmann::json;

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

    /*std::vector<Orbital> all_alpha_orbitals = alpha_orbs;
    active_alpha_orbital_indicies.resize(num_alpha_orbs);
    std::fill(active_alpha_orbital_indicies.begin(), active_alpha_orbital_indicies.end(), -1);
    frozen_occupied_alpha_orbital_indicies.clear();

    std::vector<Orbital> all_beta_orbitals = beta_orbs;
    active_beta_orbital_indicies.resize(num_beta_orbs);
    std::fill(active_beta_orbital_indicies.begin(), active_beta_orbital_indicies.end(), -1d);
    frozen_occupied_beta_orbital_indicies.clear();*/
    
    std::vector<Orbital> all_alpha_orbitals = alpha_orbs;
    std::vector<Orbital> all_beta_orbitals = beta_orbs;
    //int dim_ab = num_alpha_orbs + num_beta_orbs;
    //active_spin_orb_indicies.resize(dim_ab);
    //std::fill(active_spin_orb_indicies.begin(), active_spin_orb_indicies.end(), -1);
    frozen_occ_spin_orb_indices.clear();

    int max_dim = std::max(num_alpha_orbs, num_beta_orbs);
    
    /*for(int i = 0; i < max_dim; ++i) {
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
    }*/

    num_active_alpha = 0;
    num_active_beta = 0;
    
    for(int i = 0; i < max_dim; ++i) {
        real_function_3d orb = real_factory_3d(*world);
        if (i < num_alpha_orbs) {
            load(orb, all_alpha_orbitals[i].orbital_file_name);
            all_alpha_orbitals[i].function = orb;
            if(all_alpha_orbitals[i].orbital_type == "active") {
                active_spin_orb_indices.push_back(all_alpha_orbitals[i].active_space_index);
                active_alpha_beta_orbs.push_back(all_alpha_orbitals[i].function);
                ++num_active_alpha;
            }
            else if(all_alpha_orbitals[i].orbital_type == "frozen_occupied") {
                frozen_occ_spin_orb_indices.push_back(i); //needs to be revisited
            }
        }
        if (i < num_beta_orbs) {
            load(orb, all_beta_orbitals[i].orbital_file_name);
            all_beta_orbitals[i].function = orb;
            if(all_beta_orbitals[i].orbital_type == "active") {
                active_spin_orb_indices.push_back(all_beta_orbitals[i].active_space_index);
                active_alpha_beta_orbs.push_back(all_beta_orbitals[i].function);
                ++num_active_beta;
                }
            else if(all_beta_orbitals[i].orbital_type == "frozen_occupied") {
            frozen_occ_spin_orb_indices.push_back(i);
            }
        }
    }

    //Print frozen occupied indicies
    /*std::cout << "Frozen occupied alpha orbitals: ";
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
    std::cout << std::endl;*/

    std::cout << "Frozen occupied spin orbitals: ";
    for (auto const& FroIdx : frozen_occ_spin_orb_indices)
    {
        std::cout << FroIdx << " ";
    }
    std::cout << std::endl;

    //Print active space indicies
    /*std::cout << "Active space alpha indicies: ";
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
    std::cout << std::endl;*/

    std::cout << "Active space spin orbitals indicies: ";
    for (auto const& AIdx : active_spin_orb_indices)
    {
        std::cout << AIdx << " ";
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
    npy::npy_data alpha_rdm_npy = npy::read_npy<double>(one_body_rdm_files[0]);
    npy::npy_data beta_rdm_npy = npy::read_npy<double>(one_body_rdm_files[1]);
    npy::npy_data alpha_alpha_rdm_npy = npy::read_npy<double>(two_body_rdm_files[0]);
    npy::npy_data alpha_beta_rdm_npy = npy::read_npy<double>(two_body_rdm_files[1]);
    npy::npy_data beta_beta_rdm_npy = npy::read_npy<double>(two_body_rdm_files[2]);

    std::vector<double> data_alpha = alpha_rdm_npy.data; //extract the data and recast as std::vector
    std::vector<double> data_beta = beta_rdm_npy.data;
    std::vector<double> data_alpha_alpha = alpha_alpha_rdm_npy.data;
    std::vector<double> data_alpha_beta = alpha_beta_rdm_npy.data;
    std::vector<double> data_beta_beta = beta_beta_rdm_npy.data;
    
    //int dim_alpha = active_alpha_orbital_indicies.size();
    //int dim_beta = active_beta_orbital_indicies.size();
    int max_dim = active_alpha_beta_orbs.size();

    //Insert the rdm vectors into the matrices and tensor, both are in phys notation
    
    Alpha_Rdm_Matrix = Eigen::MatrixXd::Zero(num_active_alpha, num_active_alpha);
    Beta_Rdm_Matrix = Eigen::MatrixXd::Zero(num_active_beta, num_active_beta);
    
    int x = 0;
    int y = 0;
    for (int i = 0; i < num_active_alpha; ++i) {
        for (int j = 0; j < num_active_alpha; ++j) {
            Alpha_Rdm_Matrix(i, j) = data_alpha[x++];
        }
    }

    //std::cout << "Alpha 1-rdms:" << std::endl;
    //std::cout << Alpha_Rdm_Matrix << std::endl; 

    for (int i = 0; i < num_active_beta; ++i) {
        for (int j = 0; j < num_active_beta; ++j) {
            Beta_Rdm_Matrix(i, j) = data_beta[y++];
        }
    }

    //std::cout << "Beta 1-rdms:" << std::endl;
    //std::cout << Beta_Rdm_Matrix << std::endl; 
    
    //old implementation of initialising Alpha_Beta_Rdm_Matrix already here
    
    /*int x = 0;
    int y = 0;
    Alpha_Beta_Rdm_Matrix = Eigen::MatrixXd::Zero(max_dim, max_dim);
    for (int i = 0; i < max_dim; i+=2){
        for (int j = 0; j < max_dim; j+=2) {
            Alpha_Beta_Rdm_Matrix(i, j) = data_alpha[x++];
        }
    }
    for (int i = 1; i < max_dim; i+=2){
        for (int j = 1; j < max_dim; j+=2) {
            Alpha_Beta_Rdm_Matrix(i, j) = data_beta[y++];
        }
    }*/
    

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
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = data_alpha_alpha[a++];
                }
            }    
        }
    }
    for (int i = 0; i < max_dim; i+=2) {
        for (int j = 1; j < max_dim; j+=2) {
            for (int k = 1; k < max_dim; k+=2) {
                for (int l = 0; l < max_dim; l+=2) {
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = data_alpha_beta[b++];
                    Alpha_Beta_Rdm_Tensor(j, i, k, l) = data_alpha_beta[c++];
                }
            }    
        }
    }
    for (int i = 1; i < max_dim; i+=2) {
        for (int j = 1; j < max_dim; j+=2) {
            for (int k = 1; k < max_dim; k+=2) {
                for (int l = 1; l < max_dim; l+=2) {
                    Alpha_Beta_Rdm_Tensor(i, j, l, k) = data_beta_beta[d++];
                }
            }    
        }
    }



    //std::cout << "Alpha Beta Rdms Matrix: " << endl << Alpha_Beta_Rdm_Matrix << endl;
    //std::cout << "Alpha Beta combined Rdms size: " << endl << Alpha_Beta_Rdm_Tensor.size() << endl;

    //int nonzero = 0;
    /*std::cout << "Alpha_Beta_rdms before" << std::endl;
    for (int i = 0; i < max_dim; ++i) {
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


void SpinorbOpt::TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix)
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
}

//void SpinorbOpt::CreateActiveSpaceRotationMatrix();
//idea to make this a seperate function perhaps?!

void SpinorbOpt::TransformToNObasis()
{
    std::cout << "Alpha rdm matrix before " << std::endl;
    std::cout << Alpha_Rdm_Matrix << std::endl;
    std::cout << "Beta rdm matrix before " << std::endl;
    std::cout << Beta_Rdm_Matrix << std::endl;
    
    int num_active_orbs = active_alpha_beta_orbs.size();
    //int dim_alpha = active_alpha_orbital_indicies.size();
    //int dim_beta = active_beta_orbital_indicies.size();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_alpha;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_beta;

    es_alpha.compute(Alpha_Rdm_Matrix);
    es_beta.compute(Beta_Rdm_Matrix);

    Alpha_Rotation_Matrix = es_alpha.eigenvectors().rowwise().reverse();
    Beta_Rotation_Matrix = es_beta.eigenvectors().rowwise().reverse();

    //Eigen::MatrixXd Alpha_Rotation_Matrix = Eigen::MatrixXd::Identity(num_active_alpha, num_active_alpha);
    //Eigen::MatrixXd Beta_Rotation_Matrix = Eigen::MatrixXd::Identity(num_active_beta, num_active_beta);

    //std::cout << "Alpha Rotation Matrix " << Alpha_Rotation_Matrix << std::endl;
    //std::cout << "Beta Rotation Matrix " << Beta_Rotation_Matrix << std::endl;

    TransformMatrix(&Alpha_Rdm_Matrix, Alpha_Rotation_Matrix);
    TransformMatrix(&Beta_Rdm_Matrix, Beta_Rotation_Matrix);
    
    /*std::cout << "Alpha rdms after " << std::endl;
    std::cout << Alpha_Rdm_Matrix << std::endl;
    std::cout << "Beta rdms after " << std::endl;
    std::cout << Beta_Rdm_Matrix << std::endl;*/

    //create blown up 1-body rdm and Active Space Rotation Matrix

    Alpha_Beta_Rdm_Matrix = Eigen::MatrixXd::Zero(num_active_orbs, num_active_orbs); //already in NO basis
    ActiveSpaceRotationMatrix = Eigen::MatrixXd::Zero(num_active_orbs, num_active_orbs);
    
    for (int i = 0; i < num_active_alpha; i+=1) {
        for (int j = 0; j < num_active_alpha; j+=1) {
            Alpha_Beta_Rdm_Matrix(i*2,j*2) = Alpha_Rdm_Matrix(i,j);
            ActiveSpaceRotationMatrix(i*2,j*2) = Alpha_Rotation_Matrix(i,j);
            //if (i == 0 && j == 0) {
            //    Alpha_Beta_Rdm_Matrix(i*2,j*2) = 1;
            //}
            //else if (i == 1 && j == 1) {
            //    Alpha_Beta_Rdm_Matrix(i*2,j*2) = 1;
            //}
            //else {
            //    Alpha_Beta_Rdm_Matrix(i*2, j*2) = 0;
            //}
        }
    }
    for (int i = 0; i < num_active_beta; i+=1) {
        for (int j = 0; j < num_active_beta; j+=1) {
            Alpha_Beta_Rdm_Matrix(i*2 + 1, j*2 + 1) = Beta_Rdm_Matrix(i,j);
            ActiveSpaceRotationMatrix(i*2 + 1, j*2 + 1) = Beta_Rotation_Matrix(i,j);
            //if (i == 0 && j == 0) {
            //    Alpha_Beta_Rdm_Matrix(i*2 + 1, j*2 + 1) = 1;
            //}
            //else {
            //    Alpha_Beta_Rdm_Matrix(i*2 + 1, j*2 + 1) = 0;
            //}
        }
    }

    std::cout << "Alpha Beta rdms in NO basis " << std::endl;
    std::cout << Alpha_Beta_Rdm_Matrix << std::endl;
    //std::cout << "Active space rotation matrix " << std::endl;
    //std::cout << ActiveSpaceRotationMatrix << std::endl;


    TransformTensor(&Alpha_Beta_Rdm_Tensor, ActiveSpaceRotationMatrix);
    
    //alle non-zeros anzeigen lassen
    /*for (int i = 0; i < num_active_orbs; ++i) {
        for (int j = 0; j < num_active_orbs; ++j) {
            for (int k = 0; k < num_active_orbs; ++k) {
                for (int l = 0; l < num_active_orbs; ++l) {
                    if (Alpha_Beta_Rdm_Tensor(i, j, k, l) != 0) {
                        std::cout << "Alpha_Beta_2rdms[" << i << "][" << j << "][" << k << "][" << l << "] = " 
                                  << Alpha_Beta_Rdm_Tensor(i, j, k, l) << std::endl;
                        
                    }
                }
            }
        }
    }*/

    madness::Tensor<double> T(num_active_orbs, num_active_orbs);
    for (int i = 0; i < num_active_orbs; i++) {
        for (int j = 0; j < num_active_orbs; j++) {
            T(i,j) = ActiveSpaceRotationMatrix(i,j);
        }
    }
    
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

    /*for (int i = 0; i < num_active_orbs; i++ ) {
        double occ = abs(Alpha_Beta_Rdm_Matrix(i, i));
        std::cout << "Occupation of Orbital " << i << " is " << occ << std::endl; //wenn occ fast eins, ist orbital besetzt -> kriterium fur n_alpha und n_beta
    }*/

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

    Eigen::MatrixXd overlap_matrix_alpha = Eigen::MatrixXd::Zero(num_active_alpha, num_active_alpha);
    Eigen::MatrixXd overlap_matrix_beta = Eigen::MatrixXd::Zero(num_active_beta, num_active_beta);

    //Calculate the one-body integrals
    // <alpha|alpha> elements 
    
    for(int k = 0; k < dim_ab; k+=2) {
        for (int l = 0; l < dim_ab; l+=2) {
        //std::cout << k << std::endl;
        //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_ab = free_space_derivative<double,3>(*world, axis);
                real_function_3d dab_orb_left = D_ab(active_alpha_beta_orbs[k]);
                real_function_3d dab_orb_right = D_ab(active_alpha_beta_orbs[l]);
                integrals_kinetic[axis](k, l) = 0.5 * inner(dab_orb_left,dab_orb_right);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(active_alpha_beta_orbs[l]);
            integrals_potential(k, l) = inner(active_alpha_beta_orbs[k], Vnuc_orb_l);
            }
        }

    //<beta|beta> elements

    for(int k = 1; k < dim_ab; k+=2) {
        for (int l = 1; l < dim_ab; l+=2) {
            //std::cout << k << std::endl;
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_ab = free_space_derivative<double,3>(*world, axis);
                real_function_3d dab_orb_left = D_ab(active_alpha_beta_orbs[k]);
                real_function_3d dab_orb_right = D_ab(active_alpha_beta_orbs[l]);
                integrals_kinetic[axis](k, l) = 0.5 * inner(dab_orb_left,dab_orb_right);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(active_alpha_beta_orbs[l]);
            integrals_potential(k, l) = inner(active_alpha_beta_orbs[k], Vnuc_orb_l);
        }
    }

    integrals_one_body = integrals_potential + integrals_kinetic[0] + integrals_kinetic[1] + integrals_kinetic[2];

    std::cout << "One body integrals: " << integrals_one_body << std::endl;
    
    for (int i = 0; i < num_active_alpha; i++) {
        for(int j = 0; j < num_active_alpha; j++) {
            overlap_matrix_alpha(i, j) = inner(active_alpha_beta_orbs[2*i], active_alpha_beta_orbs[2*j]);
        }
    }
    
    std::cout << "Alpha Overlap matrix: " << std::endl;
    std::cout << overlap_matrix_alpha << std::endl;

    for (int i = 0; i < num_active_beta; i++) {
        for(int j = 0; j < num_active_beta; j++) {
            overlap_matrix_beta(i, j) = inner(active_alpha_beta_orbs[2*i + 1], active_alpha_beta_orbs[2*j + 1]);
        }
    }
    
    std::cout << "Beta Overlap matrix: " << std::endl;
    std::cout << overlap_matrix_beta << std::endl;

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

    std::cout << "mn size: " << ab_coul_orbs_mn.size() << std::endl;

    //calculate and insert the non-zero elements to the two-body integral tensor, in phys notation already

    madness::Tensor<double> Inner_prods = matrix_inner(*world, ab_orbs_kl, ab_coul_orbs_mn, false);

    //<alpha alpha|alpha alpha> elements
    for(int k = 0; k < dim_ab; k+=2)
    {
        for(int l = 0; l < dim_ab; l+=2)
        {
            for(int m = 0; m < dim_ab; m+=2)
            {
                for(int n = 0; n < dim_ab; n+=2)
                {
                    integrals_two_body(k, m, l, n) = Inner_prods(k*dim_ab + l, m*dim_ab + n);
                }
            }
        }    
    }   
    
    //<beta beta|beta beta> elements
    
    for(int k = 1; k < dim_ab; k+=2)
    {
        for(int l = 1; l < dim_ab; l+=2)
        {
            for(int m = 1; m < dim_ab; m+=2)
            {
                for(int n = 1; n < dim_ab; n+=2)
                {
                    integrals_two_body(k, m, l, n) = Inner_prods(k*dim_ab + l, m*dim_ab + n);
                }
            }
        }    
    }
    
    //<alpha beta |alpha beta> elements
    //<beta alpha|beta alpha> elements
    
    for(int k = 0; k < dim_ab; k+=2)
    {
        for(int l = 0; l < dim_ab; l+=2)
        {
            for(int m = 1; m < dim_ab; m+=2)
            {
                for(int n = 1; n < dim_ab; n+=2)
                {
                    integrals_two_body(k, m, l, n) = Inner_prods(k*dim_ab + l, m*dim_ab + n);
                    integrals_two_body(m, k, n, l) = Inner_prods(m*dim_ab + n, k*dim_ab + l);
                }
            }
        }    
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateAllIntegrals took " << duration.count() << " seconds" << std::endl;

    //std::cout << "Two body integrals size " << integrals_two_body.size() << std::endl;
 
    //int nonzero = 0;
    /*for (int i = 0; i < dim_ab; i++) {
        for (int j = 0; j < dim_ab; j++) {
            for (int k = 0; k < dim_ab; k++) {
                for (int l = 0; l < dim_ab; l++) {
                    if (integrals_two_body(i, j, k, l) != 0) {
                        //nonzero +=1;
                        std::cout << "Alpha_Beta_integrals[" << i << "][" << j << "][" << k << "][" << l << "] = " 
                                  << integrals_two_body(i, j, k, l) << std::endl;
                    }
                }
            }
        }
    }*/
    //std::cout << "Nonzero elements: " << nonzero << std::endl;*/
}


std::vector<double> SpinorbOpt::CalculateEnergy()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> energies;

    int dim_ab = Alpha_Beta_Rdm_Matrix.rows();

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
    energies.push_back(kinetic_energy);
    energies.push_back(nuclear_attraction_energy);
    
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

    energies.push_back(two_electron_energy);

    double total_energy = kinetic_energy + two_electron_energy + nuclear_attraction_energy + nuclear_repulsion_energy;

    energies.push_back(nuclear_repulsion_energy);
    energies.push_back(total_energy);

    print("            Kinetic energy ", kinetic_energy);
    print(" Nuclear attraction energy ", nuclear_attraction_energy);
    print("       Two-electron energy ", two_electron_energy);
    print(" Nuclear  repulsion energy ", nuclear_repulsion_energy);
    print("              Total energy ", total_energy);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateEnergies took " << duration.count() << " seconds" << std::endl;

    return energies;
}

void SpinorbOpt::CalculateLagrangeMultiplier()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int dim = active_alpha_beta_orbs.size();
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
    std::cout << "Lagrange Multiplier: " << LagrangeMultiplier << std::endl;
}

double SpinorbOpt::CalculateLagrangeMultiplierElement(int dim, int a, int i)
{
    double element = Alpha_Beta_Rdm_Matrix(i,i) * integrals_one_body(a, i);

    for(int l = 0; l < dim; l++){
        for(int n = 0; n < dim; n++){
            for(int k = 0; k < dim; k++){
                element += Alpha_Beta_Rdm_Tensor(k, l, i, n) * integrals_two_body(a, l, k, n);
            }
        }
    }
    return element;
}


void SpinorbOpt::OptimizeSpinorbitals(double optimization_thresh, double NO_occupation_thresh, std::string OutputPath, std::string rdm_iteration)
{
    
    
    int num_active_orbs = active_alpha_beta_orbs.size();
    //ignore frozen orbitals for now
    /*std::vector<real_function_3d> frozen_orbs; 
    for (auto const& frozenIdx : frozen_occ_spin_orb_indicies)
    {
        frozen_orbs.push_back(all_orbitals[frozenIdx].function);
    }*/
    
    std::vector<real_function_3d> orbitals;
    for (int i = 0; i < num_active_orbs; i++)
    {
        orbitals.push_back(active_alpha_beta_orbs[i]);
    }
    

    bool converged = false;
    int iterstep = 0;
    std::cout << "Calculate initial energies: " << std::endl;
    std::vector<double> initial_energies = CalculateEnergy();
    std::cout << "Calculate initial total spin: " << std::endl;
    double initial_spin_squared = CalculateTotalSpin();

    //Save the energetic values in a json file
    json energies_array = json::array();
    json initial_energies_forjson;

    initial_energies_forjson["Iteration"] = iterstep;
    initial_energies_forjson["Kinetic energy"] = initial_energies[0];
    initial_energies_forjson["Nuclear attraction energy"] = initial_energies[1];
    initial_energies_forjson["Two-electron energy"] = initial_energies[2];
    initial_energies_forjson["Nuclear repulsion energy"] = initial_energies[3];
    initial_energies_forjson["Total energy"] = initial_energies[4];
    initial_energies_forjson["Spin_squared"] = initial_spin_squared;
    energies_array.push_back(initial_energies_forjson);
    
    while(!converged)
    {
        iterstep++;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;

        //Update orbitals
       // auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_error = 0;

        //Evaluate which orbitals are going to be updated
        std::vector<int> spin_orbs_indices_for_update;
        for (int idx = 0; idx < num_active_orbs; idx++) {
            if(abs(Alpha_Beta_Rdm_Matrix(idx, idx)) >= NO_occupation_thresh) {
                spin_orbs_indices_for_update.push_back(active_spin_orb_indices[idx]);
                }
            else {
                std::cout << "Skip optimization of spin orbital " << active_spin_orb_indices[idx] << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")" << std::endl;
            }    
        } 
    
        std::cout << "Spin orbital indices list for update : ";
        for (auto const& Idx : spin_orbs_indices_for_update) {
            std::cout << Idx << " ";
            }
        std::cout << std::endl;
            
        //Calculate the Lagrange Multiplier
        CalculateLagrangeMultiplier();

        auto start_orb_update_time = std::chrono::high_resolution_clock::now();

        std::vector<real_function_3d> AllActiveSpinOrbitalUpdates = GetAllActiveSpinorbitalUpdates(spin_orbs_indices_for_update);
        
        for (int idx = 0; idx < spin_orbs_indices_for_update.size(); idx++)
        {
            int actIdx = spin_orbs_indices_for_update[idx];
            orbitals[actIdx] = orbitals[actIdx] - AllActiveSpinOrbitalUpdates[idx];
        }
        orbitals = truncate(orbitals, 1e-5);
      
        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        std::cout << "Orthonormalize orbitals" << std::endl;
        //Project out FrozenOrbitals; to be ignored currently
        /*if(frozen_orbs.size() > 0)
        {
            auto Q = QProjector(*world, frozen_orbs);
            for (auto const& actIdx : active_orbital_indicies)
            {
                orbitals[actIdx] = Q(orbitals[actIdx]);
            }
        }*/

        //Orthonormalize orbitals
        std::vector<real_function_3d> alpha_orbs;
        std::vector<real_function_3d> beta_orbs;
    
        for(int i = 0; i < orbitals.size(); i++) {
            if (i % 2 == 0){
                alpha_orbs.push_back(orbitals[i]);
            }
            if (i % 2 == 1) {
                beta_orbs.push_back(orbitals[i]);
            }
        }

        //std::vector<real_function_3d> alpha_orbs_project = ProjectSpinorbitals(alpha_orbs);
        //std::vector<real_function_3d> beta_orbs_project = ProjectSpinorbitals(beta_orbs);
        
        //alpha_orbs_project = orthonormalize_symmetric(alpha_orbs_project);
        //beta_orbs_project = orthonormalize_symmetric(beta_orbs_project);

        alpha_orbs = orthonormalize_symmetric(alpha_orbs);
        beta_orbs = orthonormalize_symmetric(beta_orbs);

        //Update all_orbitals
        int x = 0;
        int y = 0;
        for (int i = 0; i < num_active_orbs; i++){
            if (i % 2 == 0){
                active_alpha_beta_orbs[i] = alpha_orbs[x].truncate(1e-5);
                x++;
            }   
            if (i % 2 == 1) {
                active_alpha_beta_orbs[i] = beta_orbs[y].truncate(1e-5);
                y++;
            }
        }

        std::cout << "Update Integrals" << std::endl;
        //Update integrals for new orbitals
        CalculateAllIntegrals();

        //Calculate new energy
        std::vector<double> energies = CalculateEnergy();

        json energy_iter;

        energy_iter["Iteration"] = iterstep;
        energy_iter["Kinetic energy"] = energies[0];
        energy_iter["Nuclear attraction energy"] = energies[1];
        energy_iter["Two-electron energy"] = energies[2];
        energy_iter["Nuclear repulsion energy"] = energies[3];
        energy_iter["Total energy"] = energies[4];
        energies_array.push_back(energy_iter);

    
        //Check convergence
        std::cout << "Highest error: " << highest_error << std::endl;
        if(highest_error < optimization_thresh){
            converged = true;
            std::cout << "Convergence has been reached :-)" << std::endl;
        }
    }

    std::cout << "Calculate Spin Squared" << std::endl;
    double spin_squared = CalculateTotalSpin();

    SpinorbOpt_data["SpinorbOpt output"] = energies_array;
    SpinorbOpt_data["Spin squared of refined orbitals"] = spin_squared;

    std::cout << "Save energies to a json file" << std::endl;
    std::ofstream file(OutputPath + "/SpinorbOpt_energies_test_" + rdm_iteration + ".json");
    file << SpinorbOpt_data.dump(4);
    file.close();

    std::cout << "Transform Rdms, Integrals and Orbitals Back" << std::endl;
    TransformIntegralsAndOrbitalsBack();

    //Calculate new energy
    std::cout << "Calculate energy of integrals transformed back" << std::endl;
    std::vector<double> energies = CalculateEnergy();

}


/*std::vector<real_function_3d> SpinorbOpt::ProjectSpinorbitals(std::vector<real_function_3d> orbs) //projects SO one by one out
{
    int dim_orbs = orbs.size();
    
    for (int i = 0; i < dim_orbs-1; i++) {
        std::vector<real_function_3d>  orb_for_projection;
        orb_for_projection.push_back(orbs[i]);
        auto Q_project = QProjector(*world, orb_for_projection);
        
        for (int q = i+1; q < dim_orbs; q++) {
            orbs[q] = Q_project(orbs[q]);
        }
        orb_for_projection.clear();
    }
  
    return orbs;
}*/


double SpinorbOpt::CalculateTotalSpin()
{
    double diag_alpha = 0;
    double diag_beta = 0;

    for (int i = 0; i < num_active_alpha; i++) {
        diag_alpha += Alpha_Rdm_Matrix(i,i);
    }

    //std::cout << "trace of Alpha rdm: " << diag_alpha << std::endl;

    for (int j = 0; j < num_active_beta; j++) {
        diag_beta += Beta_Rdm_Matrix(j,j);
    }

    //std::cout << "trace of Beta rdm: " << diag_beta << std::endl;
    
    double spin_squared = 0.25 * (diag_alpha - diag_beta) * (diag_alpha - diag_beta);
    spin_squared += 0.5 * (diag_alpha + diag_beta);

    //std::cout << "Sz part: " << spin_squared << std::endl;

    Eigen::MatrixXd overlap_matrix_ab = Eigen::MatrixXd::Zero(num_active_alpha, num_active_beta);
    for (int i = 0; i < num_active_alpha; i++) {
        for(int j = 0; j < num_active_beta; j++) {
            overlap_matrix_ab(i, j) = inner(active_alpha_beta_orbs[2*i], active_alpha_beta_orbs[2*j + 1]);
        }
    }

    for (int i = 0; i < num_active_beta; i++) {
        for (int j = 0; j < num_active_alpha; j++) {
            for (int k = 0; k < num_active_alpha; k++) {
                for (int l = 0; l < num_active_beta; l++) {
                    double S_ik = overlap_matrix_ab(k, i);
                    double S_jl = overlap_matrix_ab(j, l);
                    double element = S_ik * S_jl * Alpha_Beta_Rdm_Tensor(2*i+1, 2*j, 2*l+1, 2*k);
                    spin_squared -= element;
                }
            }
        }
    }

    std::cout << "S squared: " << spin_squared << std::endl;
    return spin_squared;
    
}


std::vector<real_function_3d> SpinorbOpt::GetAllActiveSpinorbitalUpdates(std::vector<int> spin_orbs_indices_for_update)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    int dim = active_alpha_beta_orbs.size();

    std::vector<real_function_3d> AllSpinorbitalUpdates;
    int dim_update = spin_orbs_indices_for_update.size();
    
    //Calculate inverse_onebody_rdm values
    std::vector<double> rdm_ii_inv;
    for (int idx = 0; idx < dim_update; idx++)
    {
        int i = spin_orbs_indices_for_update[idx];
        double inv = 1 / Alpha_Beta_Rdm_Matrix(i,i);
        rdm_ii_inv.push_back(inv);
    }

    //1 electron Part
    for (int idx = 0; idx < dim_update; idx++)
    {
        int i = spin_orbs_indices_for_update[idx];
        real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);
        real_function_3d rhs = (*Vnuc)(active_alpha_beta_orbs[i]);
        for(int k = 0; k < dim; k++)
        {
            if(k != i)
            {
                rhs -= rdm_ii_inv[idx] * LagrangeMultiplier(k,i) * active_alpha_beta_orbs[k];
            }
        }
        AllSpinorbitalUpdates.push_back(rhs);
    }
    AllSpinorbitalUpdates = truncate(AllSpinorbitalUpdates, 1e-5);

    //2 electron Part
    //auto t2 = std::chrono::high_resolution_clock::now();

    
    for(int idx = 0; idx < dim_update; idx++) {
        int i = spin_orbs_indices_for_update[idx];
        for(int l = 0; l < dim; l++) {
            for(int n = 0; n < dim; n++) {
                for(int k = 0; k < dim; k++) {
                    AllSpinorbitalUpdates[idx] += rdm_ii_inv[idx] * ab_coul_orbs_mn[l*dim + n] * Alpha_Beta_Rdm_Tensor(k, l, i, n) * active_alpha_beta_orbs[k];
                }
            }
        }
        AllSpinorbitalUpdates = truncate(AllSpinorbitalUpdates, 1e-5);
    }
    
    //BSH part
    for (int idx = 0; idx < dim_update; idx++)
    {
        int i = spin_orbs_indices_for_update[idx];
        double en = LagrangeMultiplier(i,i) * rdm_ii_inv[idx];
        SeparatedConvolution<double,3> bsh_op = BSHOperator<3>(*world, sqrt(-2*en), 0.01, 1e-6);
        real_function_3d r = active_alpha_beta_orbs[i] + 2.0 * bsh_op(AllSpinorbitalUpdates[idx]); // the residual
        double err = r.norm2();
        std::cout << "Error of Orbital " << i << ": " << err << std::endl; 
        if(err > highest_error){highest_error = err; }
        AllSpinorbitalUpdates[idx] = r;
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    auto t41 = std::chrono::duration_cast<std::chrono::seconds>(t4 - t1);
    std::cout << "GetAllActiveSpinorbitalUpdates took " << t41.count() << " seconds" << std::endl;

    return AllSpinorbitalUpdates;
}

void SpinorbOpt::TransformIntegralsAndOrbitalsBack()
{
    int num_active_orbs = active_alpha_beta_orbs.size();
    
    //Create Alpha, Beta and Full rotation matrices
    Eigen::MatrixXd RotationMatrixAlphaBack = Alpha_Rotation_Matrix.transpose();
    Eigen::MatrixXd RotationMatrixBetaBack = Beta_Rotation_Matrix.transpose();

    /*std::cout << "Alpha Back rotation matri:" << std::endl;
    std::cout << RotationMatrixAlphaBack << std::endl;

    std::cout << "Beta Back rotation matri:" << std::endl;
    std::cout << RotationMatrixBetaBack << std::endl;*/


    Eigen::MatrixXd Alpha_Beta_RotationMatrixBack = Eigen::MatrixXd::Zero(num_active_orbs, num_active_orbs);
    for(int i = 0; i < num_active_alpha; i++)
    {
        for(int j = 0; j < num_active_alpha; j++)
        {
            Alpha_Beta_RotationMatrixBack(active_spin_orb_indices[2*i], active_spin_orb_indices[2*j]) = RotationMatrixAlphaBack(i,j);
        }
    }
    for(int i = 0; i < num_active_beta; i++)
    {
        for(int j = 0; j < num_active_beta; j++)
        {
            Alpha_Beta_RotationMatrixBack(active_spin_orb_indices[2*i+1], active_spin_orb_indices[2*j+1]) = RotationMatrixBetaBack(i,j);
        }
    }

    //TransformMatrix(&Alpha_Rdm_Matrix, RotationMatrixAlphaBack);
    //TransformMatrix(&Beta_Rdm_Matrix, RotationMatrixBetaBack);
    TransformMatrix(&Alpha_Beta_Rdm_Matrix, Alpha_Beta_RotationMatrixBack);
    TransformTensor(&Alpha_Beta_Rdm_Tensor, Alpha_Beta_RotationMatrixBack);

    std::cout << "Alpha Beta rdms transformed back " << std::endl;
    std::cout << Alpha_Beta_Rdm_Matrix << std::endl;
    
    //Transform the orbitals
    madness::Tensor<double> T(num_active_orbs, num_active_orbs);
    for (int i = 0; i < num_active_orbs; i++) {
        for (int j = 0; j < num_active_orbs; j++) {
            T(i,j) = Alpha_Beta_RotationMatrixBack(i,j);
        }
    }

    std::vector<real_function_3d> orbitals_rotate;
    for (int i = 0; i < num_active_orbs; i++)
        {
            orbitals_rotate.push_back(active_alpha_beta_orbs[i]);
        }
    orbitals_rotate = transform(*world, orbitals_rotate, T);
    
    for(int i = 0; i < num_active_orbs; i++)
    {
        active_alpha_beta_orbs[i] = orbitals_rotate[i];
    }

    //Transform Integrals
    TransformMatrix(&integrals_kinetic[0], Alpha_Beta_RotationMatrixBack);
    TransformMatrix(&integrals_kinetic[1], Alpha_Beta_RotationMatrixBack);
    TransformMatrix(&integrals_kinetic[2], Alpha_Beta_RotationMatrixBack);
    TransformMatrix(&integrals_potential, Alpha_Beta_RotationMatrixBack);
    TransformMatrix(&integrals_one_body, Alpha_Beta_RotationMatrixBack);
    TransformTensor(&integrals_two_body, Alpha_Beta_RotationMatrixBack);

}

 /*void SpinorbOpt::SaveNOs(std::string OutputPath)
{
    for(int i = 0; i < active_alpha_beta_orbs.size(); i++)
    {
        if (i % 2 == 0 ) {
            int a = i / 2
            save(active_alpha_beta_orbs[i], OutputPath + "/alpha_NO_" + std::to_string(idx)); // ohne das .00000 im filename
        }
        if (i % 2 == 1) {
        int b = 
        save(active_alpha_beta_orbs[i], OutputPath + "/beta_NO_" + std::to_string(i)); // ohne das .00000 im filename
        }
        
    }
}*/

void SpinorbOpt::SaveSpinorbitals(std::string OutputPath)
{
    for(int i = 0; i < num_active_alpha; i++)
    {
        std::string filename_alpha = "alpha_orbital_" + std::to_string(i);
        save(active_alpha_beta_orbs[2*i], OutputPath + "/" + filename_alpha);

    }
    for(int j = 0; j < num_active_beta; j++)
    {
        std::string filename_beta = "beta_orbital_" + std::to_string(j);
        save(active_alpha_beta_orbs[2*j+1], OutputPath + "/" + filename_beta); 
    }
}

void SpinorbOpt::SaveIntegralsToNumpy(std::string OutputPath)
{
    std::cout << "Save Integrals to npy files" << std::endl;
    
    int dim_ab = active_alpha_beta_orbs.size();

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

    int dim_alpha = num_active_alpha;
    int dim_beta = num_active_beta;    

    std::vector<unsigned long> one_alpha_e_ints_shape{(unsigned long)dim_alpha, (unsigned long)dim_alpha};
    const npy::npy_data<double> one_alpha_e_data{one_alpha_e_int_elements, one_alpha_e_ints_shape, false};
    npy::write_npy(OutputPath + "/alpha_hmatrix.npy", one_alpha_e_data);

    std::vector<unsigned long> one_beta_e_ints_shape{(unsigned long)dim_beta, (unsigned long)dim_beta};
    const npy::npy_data<double> one_beta_e_data{one_beta_e_int_elements, one_beta_e_ints_shape, false};
    npy::write_npy(OutputPath + "/beta_hmatrix.npy", one_beta_e_data);

    std::vector<unsigned long> two_e_integrals_aa_elements_shape{(unsigned long)dim_alpha, (unsigned long)dim_alpha, (unsigned long)dim_alpha, (unsigned long)dim_alpha};
    const npy::npy_data<double> two_alpha_alpha_e_data{two_e_integrals_aa_elements, two_e_integrals_aa_elements_shape, false};
    npy::write_npy(OutputPath + "/alpha_alpha_gtensor.npy", two_alpha_alpha_e_data);

    std::vector<unsigned long> two_e_integrals_bb_elements_shape{(unsigned long)dim_beta, (unsigned long)dim_beta, (unsigned long)dim_beta, (unsigned long)dim_beta};
    const npy::npy_data<double> two_beta_beta_e_data{two_e_integrals_bb_elements, two_e_integrals_bb_elements_shape, false};
    npy::write_npy(OutputPath + "/beta_beta_gtensor.npy", two_beta_beta_e_data);

    std::vector<unsigned long> two_e_integrals_ab_elements_shape{(unsigned long)dim_alpha, (unsigned long)dim_alpha, (unsigned long)dim_beta, (unsigned long)dim_beta};
    const npy::npy_data<double> two_alpha_beta_e_data{two_e_integrals_ab_elements, two_e_integrals_ab_elements_shape, false};
    npy::write_npy(OutputPath + "/alpha_beta_gtensor.npy", two_alpha_beta_e_data);

}
