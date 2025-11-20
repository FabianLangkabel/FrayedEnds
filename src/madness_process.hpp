#pragma once

#include "functionsaver.hpp"
#include <madness/mra/mra.h>
#include <madness/chem/molecule.h>
#include <madness/chem/molecular_functors.h>
#include <cstdio>
#include <unistd.h>
#include <tuple>

using namespace madness;

class RedirectOutput {
  public:
    RedirectOutput(const std::string& filename) {
        // Open the file for writing
        originalStdout = dup(STDOUT_FILENO);

        // Redirect stdout to a new file
        freopen(filename.c_str(), "w", stdout);
    }

    ~RedirectOutput() {
        // Restore the original stdout
        fflush(stdout);
        dup2(originalStdout, STDOUT_FILENO);
        close(originalStdout);
    }

  private:
    int originalStdout;
};

class MolecularGeometry; // Forward declaration

template <std::size_t NDIM> class MadnessProcess {
  public:
    World* world;
    double L;
    long k;
    double thresh;
    int initial_level;
    int truncate_mode;
    bool refine;
    int n_threads;

    MadnessProcess(double L, long k, double thresh, int initial_level = 3, int truncate_mode = 1, bool refine = true,
                   int n_threads = -1);
    ~MadnessProcess();

    void change_nthreads(int n_threads);

    // load a function from a SavedFct object
    Function<double, NDIM> loadfct(const SavedFct<NDIM>& Sf);

    // load a function from a binary file
    Function<double, NDIM> loadfct_from_file(const std::string& filename);

    std::tuple<double, long, double, int, int, bool, int> get_function_defaults();

    void update_function_defaults();

    std::tuple<double, long, double, int, int, bool, int> get_function_defaults();

    void update_function_defaults();

    void plot(std::string filename, SavedFct<NDIM> f, std::string axis, int datapoints);

    void plane_plot(std::string filename, SavedFct<NDIM> f, std::string plane, double zoom, int datapoints,
                    std::vector<double> origin);

    void cube_plot(std::string filename, SavedFct<3> f, MolecularGeometry molecule, double zoom, int datapoints,
                   std::vector<double> origin); // only defined for NDIM=3
};
