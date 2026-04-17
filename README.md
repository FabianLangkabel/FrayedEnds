<div style="display: flex; justify-content: center;">
  <img src="frayed_ends.png" width="25%">
</div>


# FrayedEnds 

FrayedEnds is a framework that discretizes continuous quantum systems and transforms their continuous Hamiltonians into an optimal second quantized form through adaptive orbital optimization. Fast and accurate calculations are performed using a Multi-Resolution-Analysis representation (implemented with [MADNESS](https://github.com/m-a-d-n-e-s-s/madness)).

# Installation
FrayedEnds supports macOS and linux.
The setup process depends on the operating system. 

## Prerequisites
Before installing FrayedEnds, make sure you have the following dependencies installed on your system:
- `CMake` (version 3.12 or higher)
- `Python` (version 3.10 or higher)
- A C++ compiler (e.g., `g++` or `clang++`) 

## Linux

On linux systems, the recommended way to install and run the project is by using the provided devcontainer. 

- The included devcontainer automatically installs all necessary system packages, **madness**, **conda** and all necessary python packages to compile the code and run it in combination with **Tequila(VQE)** or **Block2(DMRG)**.
- Information about development containers and their installation in combination with VSCode can be found at: [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers).

Alternatively, you can also install the dependencies manually. The procedure is identical to the macOS installation described below.

## MacOS
On macOS systems, the dependencies must be installed manually.

### Step 1: Install madness
Before installing **MADNESS**, make sure that **CMake** is installed on your system.

```bash
# create installation directory
mkdir madness
MADNESS_DIR=$(pwd)/madness
cd madness

# clone MADNESS repository into madness_source
git clone https://github.com/m-a-d-n-e-s-s/madness.git madness_source

# create build directory
mkdir madness_build

# configure build with CMake
cmake -D CMAKE_INSTALL_PREFIX=$MADNESS_DIR -DENABLE_MPI=OFF -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" -S madness_source -B madness_build

# compile and install
make -C madness_build -j8
cmake --build madness_build/ --target install -j8
```

### Step 2: Install FrayedEnds 
```bash
MADNESS_DIR=$MADNESS_DIR pip install -e .
```

# Examples & Tutorials

For hands-on examples and tutorials on how to use FrayedEnds, please visit the website:

--> <u> **[FrayedEnds Documentation](https://fabianlangkabel.github.io/FrayedEnds)**</u>  <-- 
