
#pragma once


#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cstdio>

using namespace madness;


//helper function to delete the used files
inline void delete_file(const std::string& filename) {
    if (std::remove(filename.c_str()) != 0) {
        throw std::runtime_error("Failed to delete file: " + filename);
    }
}

//Helper function to convert a binary .00000 file into a string
inline std::string read_binary_file(const std::string& filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the file contents into a string
    std::ostringstream buffer;
    buffer << file.rdbuf();

    // Close the file
    file.close();

    // Return the contents as a string
    return buffer.str();
}

//This class is used to save the MRA function
//The constructor is called with the respective MRA function and the MRA data is saved in the saved_str member
class SavedFct {
    public:
        std::string saved_str=""; //should this be private?
        std::string info="";

        SavedFct(Function<double,3> f) {
            std::string filename = "saved_fct"; //TO DO: make sure this filename is unique
            save(f,filename);
            saved_str = read_binary_file(filename+".00000");
            delete_file(filename+".00000");
        }

};



//Helper function to convert a string into a binary .00000 file
inline void write_binary_file(const SavedFct& Sf, const std::string& filename) { //TODO: make sure the filename is unique
    // Open the file in binary mode
    std::string content=Sf.saved_str;
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write the string content to the file
    file.write(content.data(), content.size());

    // Close the file
    file.close();
}

/*
class MadnessProcess {
    public:
        //Constructor
        MadnessProcess(double L, long k, double thresh) : thresh(thresh), L(L){
            int arg = 0;
            char **a = new char*[0]();

            world = &initialize(arg, a);
            startup(*world,arg,a);
            delete[] a;

            FunctionDefaults<3>::set_k(k);
            FunctionDefaults<3>::set_thresh(thresh);
            FunctionDefaults<3>::set_cubic_cell(-L,L);
        }
        //Destructor
        ~MadnessProcess() {
            std::cout << "Finalize madness env" << std::endl;
            world->gop.fence();
            finalize(); 
        }
        //Plotting function
        void plot(const char* filename, const Function<double,3>& f) {
            coord_3d lo(0.0), hi(0.0);
            lo[2] = -L; hi[2] = L;
            plot_line(filename,401,lo,hi,f);
        }
        //load a function from a SavedFct object
        Function<double,3> loadfct(const SavedFct& Sf) {
            std::string filename = "saved_fct2.00000";
            write_binary_file(Sf,filename);
            Function<double,3> f1 = real_factory_3d(*world);
            load(f1,filename);
            delete_file(filename+".00000");
            return f1;
        }

        
        private:
            World* world;
            double L;
            double thresh;
};






NB_MODULE(functionsaver, m) {
    nb::class_<real_function_3d>(m,"real_function_3d")
    .def(nb::init<>());

    nb::class_<MadnessProcess>(m, "MadnessProcess")
    .def(nb::init<const double &, const int &, const double &>())
    .def("get_guess", &MadnessProcess::get_guess)
    .def("loadfunction", &MadnessProcess::loadfunction)
    .def("plot", &MadnessProcess::plot);

    nb::class_<SavedFct>(m, "SavedFct")
    .def(nb::init<const Function<double,3> &>());
}
*/