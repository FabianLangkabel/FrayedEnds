### Attention: This code is a first version that may still contain various bugs and has not yet been cleaned up/optimized.


- The included devcontainer automatically installs all necessary system packages, madness, conda and all necessary python packages to compile the code and run it in combination with Tequila(VQE) or Block2(DMRG).
- Information about development containers and their installation in combination with VSCode can be found at: [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Installation / Compilation
MRA-OrbitalOptimization
This repository implements the OrbitalOptimization Code described in (https://arxiv.org/pdf/2410.19116) and the theory section. The orbitals are optimized using MADNESS (https://github.com/m-a-d-n-e-s-s/madness) and the variational quantum eigensolver (VQE) part is handled by tequila (https://github.com/tequilahub/tequila) with a qulacs(https://github.com/qulacs/qulacs) backend. The functions which create initial guess orbitals and optimize them are written in c++ and exposed in python using nanobind (https://github.com/wjakob/nanobind).

Installation Instructions
1. Clone the Repository
git clone https://github.com/FabianLangkabel/MRA-OrbitalOptimization.git
cd MRA-OrbitalOptimization
git checkout ts-nanobind

2. Recommended for Linux: Use the Provided Docker Environment
A Dockerfile is provided for a reproducible setup. This will install all dependencies, including MADNESS, Intel MKL, PySCF, Tequila, and nanobind. Just open the project in VSCode and choose to install the devcontainer.
Information about development containers and their installation in combination with VSCode can be found at: https://code.visualstudio.com/docs/devcontainers/containers.

After setting up the Devcontainer run in the MRA-OrbitalOptimization folder:

mkdir build
cd build
cmake ..
make

Make sure to change the absolute paths in sys.path.append() in the MRA-OrbitalOptimization/Examples/Full_run/VQE/*/run.py to match your installation of the project.

Unfortunately the Dockerfile doesn’t work for Mac:3. Manual Installation for Mac:
Requirements: cmake, miniconda/anaconda
Install MADNESS:
git clone https://github.com/m-a-d-n-e-s-s/madness.git
cd madness
mkdir build
cd build
cmake -DENABLE_MPI=OFF -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" ..
make -j8
Set up conda env:

conda create -n MRAOOenv python=3.11
conda activate MRAOOenv
pip install pyscf
pip install tequila-basic
pip install qulacs
pip install block2
python -m pip install nanobind


After activating the new conda env:
Install MRA-OrbitalOptimization

git clone https://github.com/FabianLangkabel/MRA-OrbitalOptimization.git 
cd MRA-OrbitalOptimization
git checkout ts_nanobind 
mkdir build 
cd build 
cmake .. 
make -j8


## Execution of examples

1. H₂ Example
Location:
MRA-OrbitalOptimization/Examples/Full_run/VQE/H2 


How to Run:
Execute the run.py file

What does it do:
gets a molecule geometry in bohr units (geometry_bohr) as well as parameters such as the number of iterations of the code and the parameters for the PNO and Orbital-Optimization calculations which are done using MADNESS. 
Geometry is transformed to Ångström units for tequila inputs.
The number of frozen and active orbitals is specified.
Start of the PNO calculation:
The output is redirected to a file “PNO.out” using a RedirectOutput object. This object changes the stdout of the C++ and python code. To change the output back, delete the RedirectOutput object.The PNO calculation is initialized with a ParametersQC object, which is created using the geometry data. basis_set and multiplicity have to be set to None and 1, respectively. After creating a molecule file, which the PNO code needs in order to read the geometry, the PNO interface is initialized. Since this is a MADNESS process, box_size, wavelet_order, madness_thresh are required. The name of the molecule file (“molecule”) and dft={“L":box_size} are also required. After this the main PNO algorithm is executed. This code creates initial guess orbitals, the h and g tensors and the nuclear repulsion c. These results can be obtained using the Get… functions. After every finished MADNESS process the object which initializes the MADNESS process needs to be deleted!!! Otherwise your code will crash.PNO_cleanup deletes temporary files created by the PNO calc.
With the initial guess orbitals, the h and g tensors and c, we can start the main algorithm. First: VQE:
To start the VQE a molecule “mol” is initialized using tq.Molecule. tq.Molecule needs the geometry of the molecule in Ångström, and h, g and c. After this the VQE algorithm is set up. 
In the first iteration step, the function get_best_initial_values is used to get an initial guess for the VQE, afterwards the values of the previous iteration are used. The results of the VQE are the one and two body reduced density matrizes rdm1 and rdm2. 

Second: Orbital Optimimzation

The orbital optimization process is handled by the Optimization class. After initializing the class with the standard MADNESS parameters box_size, wavelet_order and madness_thresh, as well as setting some parameters the orbitals (all_orbs) and the rdms are set. With this input the orbital optimization algorithm is executed. The results of the algorithm are new orbitals, new tensors h and g and a (potentially) different nuclear repulsion c. A line plot along the z-axis of the orbitals is saved to orbital_*.dat files using the Optimization.plot() function. 

With these results the algorithm is started again, until all iterations are done. 


2. Custom Potential Example
Location:run.py
Description:This example extends the H₂ workflow by introducing a custom potential for the electronic structure calculation. The custom potential is created as the Coulomb potential of a charge density. This charge density is a sum of Gaussians such that rho(r)=\sum_i e^(-a_i*(r-r_i)^2), where a_i and r_i are specified in the lists sharpness_list and peak_loc, respectively. Q is the total charge of the charge density. The rest of the algorithm works exactly the same, except that instead of initializing a nuclear potential in the orbital optimization step (using CreateNuclearPotentialAndRepulsion(“molecule”)), we initialize the algorithm with GiveCustomPotential(custom_pot). 
If you want to try different custom potentials, make sure the molecular geometry approximately matches the charge distribution, since the initial guess orbitals are created using the molecule. Furthermore, the amount of electrons in your system is also determined by the molecular geometry, so modify the atoms to modify the number of electrons. 

