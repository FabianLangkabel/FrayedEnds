#pragma once

#include "functionsaver.hpp"
#include <madness/mra/mra.h>
#include <cstdio>
#include <unistd.h>

using namespace madness;

class RedirectOutput{
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

class MadnessProcess {
    public:
        World* world;
        MadnessProcess(double L, long k, double thresh, int initial_level=5, int truncate_mode=1, bool refine=true){
            int arg = 0;
            char **a = new char*[0]();

            world = &initialize(arg, a);
            startup(*world,arg,a);
            delete[] a;
            

            FunctionDefaults<3>::set_k(k);
            FunctionDefaults<3>::set_thresh(thresh);
            FunctionDefaults<3>::set_refine(refine);
            FunctionDefaults<3>::set_initial_level(initial_level);
            FunctionDefaults<3>::set_truncate_mode(truncate_mode);
            FunctionDefaults<3>::set_cubic_cell(-L, L);
        }
        ~MadnessProcess(){
            std::cout << "Finalize madness env" << std::endl;
            world->gop.fence();
            finalize();
        }
        //load a function from a SavedFct object
        Function<double,3> loadfct(const SavedFct& Sf){
            std::string filename = "saved_fct2"; //TODO: check if filename is unique
            write_binary_file(Sf,filename);
            Function<double,3> f1 = real_factory_3d(*world);
            load(f1,filename);
            delete_file(filename+".00000");
            return f1;
        }
        //load a function from a binary file
        Function<double,3> loadfct_from_file(const std::string& filename) {
            Function<double,3> f1 = real_factory_3d(*world);
            load(f1,filename);
            return f1;
        }

        void plot(std::string filename, SavedFct f, int datapoints=2001){
            Vector<double,3> lo(0.0), hi(0.0);
            double L = FunctionDefaults<3>::get_cell_width()[0]/2;
            lo[2] = -L; hi[2] = L;
            plot_line(filename.c_str(),datapoints,lo,hi,loadfct(f));
        }
};


