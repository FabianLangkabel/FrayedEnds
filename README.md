### Attention: This code is a first version that may still contain various bugs and has not yet been cleaned up/optimized.

## Installation / Compilation
- The included devcontainer automatically installs all necessary system packages, madness, conda and all necessary python packages to compile the code and run it in combination with Tequila(VQE) or Block2(DMRG).
- Information about development containers and their installation in combination with VSCode can be found at: [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Execution of examples
- In order to execute the script, the spin orbital refinement code (spinorb_optimizer.cpp) must first be compiled in the /workspaces/MRA-OrbitalOptimization/build folder. Then the script can be executed from the terminal with a json file, at the moment only two examples are provided. The command is either 
/workspaces/MRA-OrbitalOptimization/build/madness_programs/spinorb_opt/SpinorbOpt /workspaces/MRA-OrbitalOptimization/Json_files/input_spinorbt/h2_sto3g_inp.json 
or 
/workspaces/MRA-OrbitalOptimization/build/madness_programs/spinorb_opt/SpinorbOpt /workspaces/MRA-OrbitalOptimization/Json_files/input_spinorbt/h3lin_sto3g_inp.json 
depending on the json file selected.