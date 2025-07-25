#pragma once

#include "functionsaver.hpp"
#include <madness/mra/mra.h>
#include <madness/chem/molecule.h>
#include <madness/chem/molecular_functors.h>
#include <cstdio>
#include <unistd.h>

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

class MadMolecule; // Forward declaration
class MadnessProcess {
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
    Function<double, 3> loadfct(const SavedFct& Sf);

    // load a function from a binary file
    Function<double, 3> loadfct_from_file(const std::string& filename);

    void plot(std::string filename, SavedFct f, std::string axis = "z", int datapoints = 2001);

    void plane_plot(std::string filename, SavedFct f, std::string plane = "yz", double zoom = 1.0, int datapoints = 151,
                    std::vector<double> origin = {0.0, 0.0, 0.0});

    void cube_plot(std::string filename, SavedFct f, MadMolecule molecule, double zoom = 1.0, int datapoints = 151,
                   std::vector<double> origin = {0.0, 0.0, 0.0});
};

class MadMolecule {
  public:
    MadMolecule();

    void add_atom(float pos_x, float pos_y, float pos_z, std::string symbol);

    std::string to_json();

    SavedFct compute_nuclear_derivative(MadnessProcess& mp, const int atom, const int axis);

    madness::Molecule mol;
};
