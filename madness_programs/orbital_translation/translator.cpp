#include "translator.hpp"

using namespace madness;


Translator::Translator(int argc, char** argv, double L, long k, double thresh)
{
    int arg = 0;
    char **a = new char*[0]();
    initialize(arg, a);
    world = new World(SafeMPI::COMM_WORLD);
    //this->atoms = atoms;

    startup(*world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<3>::set_k(k); //Anzahl Polynome
    FunctionDefaults<3>::set_thresh(thresh); // Genauigkeit
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L); //Wie Groß Würfel -50 -> 50
}

Translator::~Translator()
{

}

void Translator::create_aos(std::string GeometryFile, std::string basis_set)
{
    //Define AO BasisSet
    AtomicBasisSet aobasis;
    aobasis = AtomicBasisSet();
    aobasis.read_file(basis_set);

    //Define Molecule
    molecule = madness::Molecule();
    molecule.read_file(GeometryFile);

    // Make at_to_bf, at_nbf ... map from atom to first bf on atom, and nbf/atom
    std::vector<int> at_to_bf, at_nbf;
    aobasis.atoms_to_bfn(molecule, at_to_bf, at_nbf);
    ao = vecfuncT(aobasis.nbf(molecule));

    //Project AO basis
    for (int i = 0; i < aobasis.nbf(molecule); ++i) {
        functorT aofunc(new AtomicBasisFunctor(aobasis.get_atomic_basis_function(molecule, i)));
        ao[i] = factoryT(*world).functor(aofunc).truncate_on_project().nofence().truncate_mode(1);
    }
    world->gop.fence();
    truncate(*world, ao);
    normalize(*world, ao);
}

void Translator::read_mo_coeffs(std::string calpha_coeff_file, std::string cbeta_coeff_file) 
//Reading the coefficients from .npy files
{  
    npy::npy_data d_alpha = npy::read_npy<double>(calpha_coeff_file); //read the files
    npy::npy_data d_beta = npy::read_npy<double>(cbeta_coeff_file);

    
    std::vector<double> alpha_elements= d_alpha.data; //extract the data and recast as std::vector
    std::vector<double> beta_elements = d_beta.data;

    std::vector<unsigned long> alpha_shape = d_alpha.shape; //extract the shape of the matrices
    std::vector<unsigned long> beta_shape = d_beta.shape;

    /*for (int i = 0; i < alpha_elements.size(); i++) {
        double el = alpha_elements[i];
        std::cout << "Alpha element [" << i << "] = " << el << std::endl; 
    }

    for (int j = 0; j < beta_elements.size(); j++) {
        double el = beta_elements[j];
        std::cout << "Beta element [" << j << "] = " << el << std::endl; 
    }*/
    
    //Reshape the coefficient vectors to proper matrices
    Alpha_Coeff_Matrix = Eigen::MatrixXd::Zero(alpha_shape[0], alpha_shape[1]);

    int a = 0;
    for (int b = 0; b < alpha_shape[0]; ++b) {
        for (int c = 0; c < alpha_shape[1]; ++c) {
            Alpha_Coeff_Matrix(b, c) = alpha_elements[a];
            a++;
        }
    }
    
    Beta_Coeff_Matrix = Eigen::MatrixXd::Zero(beta_shape[0], beta_shape[1]);

    int d = 0;
    for (int e = 0; e < beta_shape[0]; ++e) {
        for (int f = 0; f < beta_shape[1]; ++f) {
            Beta_Coeff_Matrix(e, f) = beta_elements[d];
            d++;
        }
    }

    //std::cout << "Alpha Vector: " << endl << Alpha_Coeff_Vector << endl;
    std::cout << "Alpha matrix: " << endl << Alpha_Coeff_Matrix << endl;
    //std::cout << "Beta Vector: " << endl << Beta_Coeff_Vector << endl;
    std::cout << "Beta matrix: " << endl << Beta_Coeff_Matrix << endl;
    
}

void Translator::create_mo_orbitals()
{
    
    int number_alpha_orbs = Alpha_Coeff_Matrix.rows();
    //std::cout << "#alpha orbs: " << number_alpha_orbs << endl;
    madness::Tensor<double> A(number_alpha_orbs, number_alpha_orbs);
    
    
    int number_beta_orbs = Beta_Coeff_Matrix.rows();
    //std::cout << "#beta orbs: " << number_beta_orbs << endl;
    madness::Tensor<double> B(number_beta_orbs, number_beta_orbs);
    
    
    for (int i = 0; i < number_alpha_orbs; i++) {
        for (int j = 0; j < number_alpha_orbs; j++) {
            A(i,j) = Alpha_Coeff_Matrix(i,j);
        }
    }
    for (int k = 0; k < number_beta_orbs; k++) {
        for (int l = 0; l < number_beta_orbs; l++) {
            B(k,l) = Beta_Coeff_Matrix(k,l);
        }
    }
    mo_alpha_orbs = transform(*world, ao, A);
    mo_beta_orbs = transform(*world, ao, B);//MOs (2 x machen für alpha_orbs und beta_orbs)
}

void Translator::print_mo_orbitals(std::string output_folder)
{
    for(int i = 0; i < mo_alpha_orbs.size(); i++)
    {
        std::string filename1 = "alpha_orbital_" + std::to_string(i);
        save(mo_alpha_orbs[i], output_folder + "/" + filename1); // ohne das .00000 im filename
    }
    for(int j = 0; j < mo_beta_orbs.size(); j++)
    {
        std::string filename2 = "beta_orbital_" + std::to_string(j);
        save(mo_beta_orbs[j], output_folder + "/" + filename2); // ohne das .00000 im filename
    }
}

// npy Files lesen/schreiben
// https://github.com/llohse/libnpy/tree/master