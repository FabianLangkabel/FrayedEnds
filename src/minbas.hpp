#pragma once

#include "functionsaver.hpp"
#include <iomanip>
#include <madness/mra/vmra.h>
#include <madness/chem/SCF.h>
#include <madness/chem/nemo.h>
#include <madness/chem/PNO.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <fstream>
#include <regex>
#include <utility> // For std::pair
#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "madness_process.hpp"

using namespace madness;
namespace nb = nanobind;

class MinBasProjector {
  public:
    MinBasProjector(MadnessProcess& mp, std::string argv) : madness_process(mp) {
        auto [argc, charArray] = stringToCharPointerArray(argv);
        parser = commandlineparser(argc, charArray);
        freeCharPointerArray(charArray, argc);
    }
    ~MinBasProjector() { atomicbasis.clear(); }

    void run() {
        SCF calc(*(madness_process.world), parser);
        calc.reset_aobasis("sto-3g");
        atomicbasis = calc.project_ao_basis(*(madness_process.world), calc.aobasis);
        basisname = "sto-3g";
        nuclear_repulsion = calc.molecule.nuclear_repulsion_energy();
        calc.make_nuclear_potential(*(madness_process.world));
        Vnuc = calc.potentialmanager->vnuclear();
    }

    std::vector<SavedFct> get_atomic_basis() const {
        std::vector<SavedFct> result;
        for (auto x : atomicbasis) {
            SavedFct y(x);
            result.push_back(y);
        }
        return result;
    }

    std::string get_basis_name() const { return basisname; }

    SavedFct get_nuclear_potential() { return SavedFct(Vnuc); }

    double get_nuclear_repulsion() const { return nuclear_repulsion; }

  private:
    MadnessProcess& madness_process;
    commandlineparser parser;
    std::vector<madness::Function<double, 3>> atomicbasis;
    std::string basisname;
    double nuclear_repulsion;
    madness::Function<double, 3> Vnuc;
};