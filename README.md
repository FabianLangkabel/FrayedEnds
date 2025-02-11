### Attention: This code is a first version that may still contain various bugs and has not yet been cleaned up/optimized.

## Installation / Compilation
- The included devcontainer automatically installs all necessary system packages, madness, conda and all necessary python packages to compile the code and run it in combination with Tequila(VQE) or Block2(DMRG).
- Information about development containers and their installation in combination with VSCode can be found at: [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Execution of examples
- In order to execute the examples provided (MRA-DMRG and MRA-VQE for BeH2 and Benzene), the orbital refinement code must first be compiled in the /workspaces/MRA-OrbitalOptimization/build folder. The examples can then be executed directly in their respective folders (python run.py)