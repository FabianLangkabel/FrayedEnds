/*this is a backup file of the functions for the Spinorbopt programm, where the alpha and beta matrices 
are evaluated seperately.
This only includes the functions without any headers, so for further use the functions need to copied back 
into the working script.

These functions require the following variables to be declared in the header file:

std::vector<Eigen::MatrixXd> integrals_kinetic_alpha;
std::vector<Eigen::MatrixXd> integrals_kinetic_beta;
Eigen::MatrixXd integrals_potential_alpha;
Eigen::MatrixXd integrals_potential_beta;
Eigen::MatrixXd integrals_one_body_alpha;
Eigen::MatrixXd integrals_one_body_beta;

Eigen::Tensor<double, 4> integrals_two_body_aa;
Eigen::Tensor<double, 4> integrals_two_body_bb;
Eigen::Tensor<double, 4> integrals_two_body_ab;
std::vector<real_function_3d> a_orbs_kl;
std::vector<real_function_3d> a_coul_orbs_mn; 
std::vector<real_function_3d> b_orbs_kl;
std::vector<real_function_3d> b_coul_orbs_mn;

*/

using namespace madness;

void SpinorbOpt::CalculateOneElectronIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    //real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);

    //construct the alpha and beta matrices
    std::vector<real_function_3d> alpha_orbitals;
    for (auto const& A_orb : all_alpha_orbitals)
    {
        alpha_orbitals.push_back(A_orb.function);
    }
    int dim_a = alpha_orbitals.size();

    std::vector<real_function_3d> beta_orbitals;
    for (auto const& B_orb : all_beta_orbitals)
    {
        beta_orbitals.push_back(B_orb.function);
    }
    int dim_b = beta_orbitals.size();

    //Set up the kinetic integrals matrices
    
    integrals_kinetic_alpha.clear();
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //x
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //y
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //z
    integrals_potential_alpha = Eigen::MatrixXd::Zero(dim_a, dim_a);
    
    integrals_kinetic_beta.clear();
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //x
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //y
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //z
    integrals_potential_beta = Eigen::MatrixXd::Zero(dim_b, dim_b);
    
    
    //One Particle Part for alpha
    for(int k = 0; k < dim_a; k++)
    {
        for(int l = 0; l < dim_a; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_a = free_space_derivative<double,3>(*world, axis);
                real_function_3d da_orb_k = D_a(alpha_orbitals[k]);
                real_function_3d da_orb_l = D_a(alpha_orbitals[l]);
                integrals_kinetic_alpha[axis](k, l) = 0.5 * inner(da_orb_k,da_orb_l);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(alpha_orbitals[l]);
            integrals_potential_alpha(k, l) = inner(alpha_orbitals[k], Vnuc_orb_l);
        }
    }

    //One Particle Part for beta
    for(int m = 0; m < dim_b; m++)
    {
        for(int n = 0; n < dim_b; n++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_b = free_space_derivative<double,3>(*world, axis);
                real_function_3d db_orb_m = D_b(beta_orbitals[m]);
                real_function_3d db_orb_n = D_b(beta_orbitals[n]);
                integrals_kinetic_beta[axis](m, n) = 0.5 * inner(db_orb_m,db_orb_n);
            }
            //Nuclear
            real_function_3d Vnuc_orb_n = (*Vnuc)(beta_orbitals[n]);
            integrals_potential_beta(m, n) = inner(beta_orbitals[m], Vnuc_orb_n);
        }
    }

    integrals_one_body_alpha = integrals_potential_alpha + integrals_kinetic_alpha[0] + integrals_kinetic_alpha[1] + integrals_kinetic_alpha[2];
    integrals_one_body_beta = integrals_potential_beta + integrals_kinetic_beta[0] + integrals_kinetic_beta[1] + integrals_kinetic_beta[2];
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateOneElectronIntegrals took " << duration.count() << " seconds" << std::endl;

    std::cout << "Alpha 1-body integrals: " << integrals_one_body_alpha << std::endl;
    std::cout << "Beta 1-body integrals: " << integrals_one_body_beta << std::endl; 

}

void SpinorbOpt::CalculateTwoElectronIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);

     //construct the alpha and beta matrices
     std::vector<real_function_3d> alpha_orbitals;
     for (auto const& A_orb : all_alpha_orbitals)
     {
         alpha_orbitals.push_back(A_orb.function);
     }
     int dim_a = alpha_orbitals.size();
 
     std::vector<real_function_3d> beta_orbitals;
     for (auto const& B_orb : all_beta_orbitals)
     {
         beta_orbitals.push_back(B_orb.function);
     }
     int dim_b = beta_orbitals.size();

     //Set up the two body integrals

    integrals_two_body_aa = Eigen::Tensor<double, 4>(dim_a, dim_a, dim_a, dim_a);
    integrals_two_body_bb = Eigen::Tensor<double, 4>(dim_b, dim_b, dim_b, dim_b);
    integrals_two_body_ab = Eigen::Tensor<double, 4>(dim_a, dim_b, dim_a, dim_b);


    // Initialising (kl|
    a_orbs_kl.clear();
    for(int k = 0; k < dim_a; k++)
    {
        std::vector<real_function_3d> kl = alpha_orbitals[k] * alpha_orbitals;
        a_orbs_kl.insert(std::end(a_orbs_kl), std::begin(kl), std::end(kl));
    }
    a_orbs_kl = truncate(a_orbs_kl, 1e-5);
    
    b_orbs_kl.clear();
    for(int k = 0; k < dim_b; k++)
    {
        std::vector<real_function_3d> kl = beta_orbitals[k] * beta_orbitals;
        b_orbs_kl.insert(std::end(b_orbs_kl), std::begin(kl), std::end(kl));
    }
    b_orbs_kl = truncate(b_orbs_kl, 1e-5);


    //apply Coulomb operator to get 1/r|mn)
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, 0.001, 1e-6));
    a_coul_orbs_mn = apply(*world, *coul_op_parallel, a_orbs_kl);
    a_coul_orbs_mn = truncate(a_coul_orbs_mn, 1e-5);
    b_coul_orbs_mn = apply(*world, *coul_op_parallel, b_orbs_kl);
    b_coul_orbs_mn = truncate(b_coul_orbs_mn, 1e-5);


    //Calculate the (aa|aa) integrals
    madness::Tensor<double> Alpha_Inner_prods = matrix_inner(*world, a_orbs_kl, a_coul_orbs_mn, false);
    for(int k = 0; k < dim_a; k++)
    {
        for(int l = 0; l < dim_a; l++)
        {
            for(int m = 0; m < dim_a; m++)
            {
                for(int n = 0; n < dim_a; n++)
                {
                    integrals_two_body_aa(k, l, m, n) = Alpha_Inner_prods(k*dim_a + l, m*dim_a + n);
                }
            }
        }
    }
    
    //Calculate the (bb|bb) integrals
    madness::Tensor<double> Beta_Inner_prods = matrix_inner(*world, b_orbs_kl, b_coul_orbs_mn, false);
    for(int k = 0; k < dim_b; k++)
    {
        for(int l = 0; l < dim_b; l++)
        {
            for(int m = 0; m < dim_b; m++)
            {
                for(int n = 0; n < dim_b; n++)
                {
                    integrals_two_body_bb(k, l, m, n) = Beta_Inner_prods(k*dim_b + l, m*dim_b + n);
                }
            }
        }
    }

    //Calculate the (aa|bb) integrals
    madness::Tensor<double> Alpha_Beta_Inner_prods = matrix_inner(*world, a_orbs_kl, b_coul_orbs_mn, false);
    for(int k = 0; k < dim_a; k++)
    {
        for(int l = 0; l < dim_a; l++)
        {
            for(int m = 0; m < dim_b; m++)
            {
                for(int n = 0; n < dim_b; n++)
                {
                    integrals_two_body_ab(k, l, m, n) = Alpha_Beta_Inner_prods(k*dim_a + l, m*dim_b + n);
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateTwoElectronIntegrals took " << duration.count() << " seconds" << std::endl;
    //std::cout << "Alpha Alpha tensor: " << integrals_two_body_aa << std::endl; //optional to print the tensors out
    //std::cout << "Beta Beta tensor: " << integrals_two_body_bb << std::endl;
    //std::cout << "Alpha Beta tensor: " << integrals_two_body_ab << std::endl;
}

real_function_3d SpinorbOpt::CalculateSpinorbitalUpdate(int orb_idx)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    int dim = active_alpha_beta_orbs.size();

    //Calculate inverse rdm element
    double rdm_ii_inv = 1 / Alpha_Beta_Rdm_Matrix(orb_idx,orb_idx);

    //1 electron part
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);
    real_function_3d rhs = (*Vnuc)(active_alpha_beta_orbs[orb_idx]);
    for(int k = 0; k < dim; k++)
    {
        if(k != orb_idx)
        {
                rhs -= rdm_ii_inv * LagrangeMultiplier(k,orb_idx) * active_alpha_beta_orbs[k];
        }
    }
    rhs.truncate(1e-5);

    //2 electron part
    for(int l = 0; l < dim; l++)
    {
        for(int n = 0; n < dim; n++)
        {
            for(int k = 0; k < dim; k++)
            {
                rhs += rdm_ii_inv * ab_coul_orbs_mn[l*dim + n] * Alpha_Beta_Rdm_Tensor(k, l, orb_idx, n) * active_alpha_beta_orbs[k];
            }
        }
    }
    rhs.truncate(1e-5);

    //BSH part

    double en = LagrangeMultiplier(orb_idx,orb_idx) * rdm_ii_inv;
    std::cout << "En value: " << en << std::endl;
    SeparatedConvolution<double,3> bsh_op = BSHOperator<3>(*world, sqrt(-2*en), 0.01, 1e-6);
    real_function_3d r = active_alpha_beta_orbs[orb_idx] + 2.0 * bsh_op(rhs); // the residual
    double err = r.norm2();
    std::cout << "Error of Orbital " << orb_idx << ": " << err << std::endl; 
    if(err > highest_error){highest_error = err; }
    
    
    auto t4 = std::chrono::high_resolution_clock::now();
    auto t41 = std::chrono::duration_cast<std::chrono::seconds>(t4 - t1);
    //auto t32 = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2);
    std::cout << "CalculateSpinorbitalUpdate took " << t41.count() << " seconds" << std::endl;
    //std::cout << "t32 took " << t32.count() << " seconds" << std::endl;

    return r;
}

void SpinorbOpt::OptimizeSpinorbitals_Test_stepwise_update(double optimization_thresh, double NO_occupation_thresh)
// function for doing only one iteration to test if stuff works
{
    int num_active_orbs = active_alpha_beta_orbs.size();

    std::vector<real_function_3d> orbitals;
    for (int i = 0; i < num_active_orbs; i++)
    {
        orbitals.push_back(active_alpha_beta_orbs[i]);
    }

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

    
    for (int i = 0; i < spin_orbs_indices_for_update.size(); i++)
    {
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start optimizing orbital " << i << std::endl;
    
        //Calculate the Lagrange Multiplier
        CalculateLagrangeMultiplier();

        int orb_idx = spin_orbs_indices_for_update[i];
        real_function_3d active_spinorbital_update = CalculateSpinorbitalUpdate(orb_idx);

        orbitals[orb_idx] = orbitals[orb_idx] - active_spinorbital_update;
        orbitals[orb_idx] = orbitals[orb_idx].truncate(1e-5);
        active_alpha_beta_orbs[orb_idx] = orbitals[orb_idx];

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "Update Orbital " << orb_idx << " took " << duration.count() << " seconds" << std::endl;

        std::cout << "Orthonormalize orbitals" << std::endl;

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

        std::vector<real_function_3d> alpha_orbs_project = ProjectSpinorbitals(alpha_orbs);
        std::vector<real_function_3d> beta_orbs_project = ProjectSpinorbitals(beta_orbs);
        
        alpha_orbs_project = orthonormalize_symmetric(alpha_orbs_project);
        beta_orbs_project = orthonormalize_symmetric(beta_orbs_project);

        //alpha_orbs = truncate(alpha_orbs, 1e-5);
        //beta_orbs = truncate(beta_orbs, 1e-5);


        /*for(int i = 0; i < alpha_orbs.size(); i++)
        {
            alpha_orbs[i] = alpha_orbs_project[i].truncate(1e-5);
        }
        or(int j = 0; j < beta_orbs.size(); j++)
        {
            beta_orbs[j] = beta_orbs_project[j].truncate(1e-5);
        }*/

        //Update all_orbitals
        int x = 0;
        int y = 0;
        for (int i = 0; i < num_active_orbs; i++){
            if (i % 2 == 0){
                active_alpha_beta_orbs[i] = alpha_orbs_project[x].truncate(1e-5);
                x++;
            }
            if (i % 2 == 1) {
                active_alpha_beta_orbs[i] = beta_orbs_project[y].truncate(1e-5);
                y++;
            }
        }


    std::cout << "Update Integrals" << std::endl;
    //Update integrals for new orbitals
    CalculateAllIntegrals();

    //Calculate new energy
    CalculateEnergy();

    //Check convergence
    //std::cout << "Highest error: " << highest_error << std::endl;
    }
}


void SpinorbOpt::OptimizeSpinorbitals_Simulataneous_Test(double NO_occupation_thresh)
//does one iteration of the spin orbital optimization where all orbitals are being updated simultaneously

{
    auto start_orb_update_time = std::chrono::high_resolution_clock::now();
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Start simultaneous orbital update " << std::endl;
    
    int num_active_orbs = active_alpha_beta_orbs.size();

    std::vector<real_function_3d> orbitals;
    for (int i = 0; i < num_active_orbs; i++)
    {
        orbitals.push_back(active_alpha_beta_orbs[i]);
    }

    //auto start_orb_update_time = std::chrono::high_resolution_clock::now();
    //highest_error = 0;

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

    std::vector<real_function_3d> alpha_orbs_project = ProjectSpinorbitals(alpha_orbs);
    std::vector<real_function_3d> beta_orbs_project = ProjectSpinorbitals(beta_orbs);
        
    alpha_orbs_project = orthonormalize_symmetric(alpha_orbs_project);
    beta_orbs_project = orthonormalize_symmetric(beta_orbs_project);

    //Update all_orbitals
    int x = 0;
    int y = 0;
    for (int i = 0; i < num_active_orbs; i++){
        if (i % 2 == 0){
            active_alpha_beta_orbs[i] = alpha_orbs_project[x].truncate(1e-5);
            x++;
        }   
        if (i % 2 == 1) {
            active_alpha_beta_orbs[i] = beta_orbs_project[y].truncate(1e-5);
            y++;
        }
    }

    std::cout << "Update Integrals" << std::endl;
    //Update integrals for new orbitals
    CalculateAllIntegrals();

    //Calculate new energy
    CalculateEnergy();

}

//Projects Spinorbitals out  one by one
std::vector<real_function_3d> SpinorbOpt::ProjectSpinorbitals(std::vector<real_function_3d> orbs) //projects SO one by one out
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
}