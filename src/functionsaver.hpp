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
        std::string type="";

        SavedFct(Function<double,3> f) {
            std::string filename = "saved_fct0504030201"; //TO DO: use the cloud for this
            save(f,filename);
            saved_str = read_binary_file(filename+".00000");
            delete_file(filename+".00000");
        }
};



//Helper function to convert a string into a binary .00000 file
inline void write_binary_file(const SavedFct& Sf, const std::string& filename) { //TODO: make sure the filename is unique
    std::string filename2 = filename + ".00000";
    // Open the file in binary mode
    std::string content=Sf.saved_str;
    std::ofstream file(filename2, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename2);
    }

    // Write the string content to the file
    file.write(content.data(), content.size());

    // Close the file
    file.close();
}
