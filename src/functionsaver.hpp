#pragma once

#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <madness/world/vector_archive.h>
#include <madness/world/parallel_archive.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cstdio>

using namespace madness;

// Serialize Function directly to string using ParallelOutputArchive with VectorOutputArchive
template <std::size_t NDIM>
inline std::string serialize_function_to_string(const Function<double, NDIM>& f) {
    World& world = f.world();
    
    // Create a local vector to hold serialized data
    std::vector<unsigned char> data;
    
    // Create VectorOutputArchive wrapping the data vector
    archive::VectorOutputArchive local_ar(data);
    
    // Create ParallelOutputArchive wrapping the VectorOutputArchive
    archive::ParallelOutputArchive<archive::VectorOutputArchive> par_ar(world, local_ar);
    
    // Serialize the function - this populates the data vector
    par_ar & f;
    
    // Convert vector<unsigned char> to string
    return std::string(reinterpret_cast<const char*>(data.data()), data.size());
}

// Deserialize Function from string using ParallelInputArchive with VectorInputArchive
template <std::size_t NDIM>
inline Function<double, NDIM> deserialize_function_from_string(
    World& world, const std::string& data) {
    
    // Convert string to vector
    std::vector<unsigned char> buffer(data.begin(), data.end());
    
    // Create VectorInputArchive wrapping the buffer
    archive::VectorInputArchive local_ar(buffer);
    
    // Create ParallelInputArchive wrapping the VectorInputArchive
    archive::ParallelInputArchive<archive::VectorInputArchive> par_ar(world, local_ar);
    
    // Deserialize the function
    Function<double, NDIM> f;
    par_ar & f;
    
    return f;
}

// This class is used to save the MRA function
// The constructor is called with the respective MRA function and the MRA data is saved in the saved_str member
template <std::size_t NDIM> class SavedFct {
  public:
    std::string saved_str = ""; // should this be private?
    std::string info = "";

    SavedFct(Function<double, NDIM> f) : info("None") { saved_str = serialize_function_to_string(f); }

    SavedFct(Function<double, NDIM> f, const std::string info) : info(info) {
        saved_str = serialize_function_to_string(f);
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
    }
};

template class SavedFct<2>;
template class SavedFct<3>;