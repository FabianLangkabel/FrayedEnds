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

// helper function to delete the used files
inline void delete_file(const std::string& filename) {
    if (std::remove(filename.c_str()) != 0) {
        throw std::runtime_error("Failed to delete file: " + filename);
    }
}

// Helper function to convert a binary .00000 file into a string
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

// This class is used to save the MRA function
// The constructor is called with the respective MRA function and the MRA data is saved in the saved_str member
template <std::size_t NDIM> class SavedFct {
  public:
    std::string saved_str = ""; // should this be private?
    std::string info = "";
    std::string type = "";

    SavedFct(Function<double, NDIM> f) : type("unknown"), info("None") { saved_str = get_data_string(f); }

    SavedFct(Function<double, NDIM> f, const std::string type) : type(type) { saved_str = get_data_string(f); }

    SavedFct(Function<double, NDIM> f, const std::string type, const std::string info) : type(type), info(info) {
        saved_str = get_data_string(f);
    }

    std::string get_data_string(Function<double, NDIM> f) const {
        std::string filename = "saved_fct0504030201"; // TODO: use the cloud for this
        save(f, filename);
        std::string data_string = read_binary_file(filename + ".00000");
        delete_file(filename + ".00000");
        return data_string;
    }

    SavedFct(const std::string& filepath) { load_from_file(filepath); }

    void save_to_file(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot open file for writing");

        size_t len;

        len = saved_str.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(saved_str.data(), len);

        len = info.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(info.data(), len);

        len = type.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(type.data(), len);
    }

    void load_from_file(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot open file for reading");

        size_t len;
        char* buffer;

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        buffer = new char[len];
        in.read(buffer, len);
        saved_str.assign(buffer, len);
        delete[] buffer;

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        buffer = new char[len];
        in.read(buffer, len);
        info.assign(buffer, len);
        delete[] buffer;

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        buffer = new char[len];
        in.read(buffer, len);
        type.assign(buffer, len);
        delete[] buffer;
    }
};

// Helper function to convert a string into a binary .00000 file
template <std::size_t NDIM>
inline void write_binary_file(const SavedFct<NDIM>& Sf,
                              const std::string& filename) { // TODO: make sure the filename is unique
    std::string filename2 = filename + ".00000";
    // Open the file in binary mode
    std::string content = Sf.saved_str;
    std::ofstream file(filename2, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename2);
    }

    // Write the string content to the file
    file.write(content.data(), content.size());

    // Close the file
    file.close();
}

template class SavedFct<2>;
template class SavedFct<3>;