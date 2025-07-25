#include "madness_process.hpp"
#include "madmolecule.hpp"

using namespace madness;

MadnessProcess::MadnessProcess(double L, long k, double thresh, int initial_level, int truncate_mode, bool refine,
                               int n_threads)
    : L(L), k(k), thresh(thresh), initial_level(initial_level), truncate_mode(truncate_mode), refine(refine),
      n_threads(n_threads) {
    int arg = 0;
    char** a = new char*[0]();

    if (n_threads == -1) { // default value is -1
        world = &initialize(arg, a);
    } else {
        world = &initialize(arg, a, n_threads);
    }

    startup(*world, arg, a);
    delete[] a;

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(refine);
    FunctionDefaults<3>::set_initial_level(initial_level);
    FunctionDefaults<3>::set_truncate_mode(truncate_mode);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

MadnessProcess::~MadnessProcess() {
    std::cout << "Finalize madness env" << std::endl;
    world->gop.fence();
    finalize();
}

void MadnessProcess::change_nthreads(int n_threads) {
    ThreadPool::end();
    ThreadPool::begin(n_threads);
    std::cout << "Changed number of threads to " << ThreadPool::size() << std::endl;
}

// load a function from a SavedFct object
Function<double, 3> MadnessProcess::loadfct(const SavedFct& Sf) {
    std::string filename = "saved_fct01020304"; // TODO: use the cloud for this
    write_binary_file(Sf, filename);
    Function<double, 3> f1 = real_factory_3d(*world);
    load(f1, filename);
    delete_file(filename + ".00000");
    return f1;
}
// load a function from a binary file
Function<double, 3> MadnessProcess::loadfct_from_file(const std::string& filename) {
    Function<double, 3> f1 = real_factory_3d(*world);
    load(f1, filename);
    return f1;
}

void MadnessProcess::plot(std::string filename, SavedFct f, std::string axis, int datapoints) {
    Vector<double, 3> lo(0.0), hi(0.0);
    double L = FunctionDefaults<3>::get_cell_width()[0] / 2;
    if (axis == "x") {
        lo[0] = -L;
        hi[0] = L;
    } else if (axis == "y") {
        lo[1] = -L;
        hi[1] = L;
    } else if (axis == "z") {
        lo[2] = -L;
        hi[2] = L;
    } else {
        std::cout << "Axis to plot not recognized." << std::endl;
        return;
    }
    plot_line(filename.c_str(), datapoints, lo, hi, loadfct(f));
}

void MadnessProcess::plane_plot(std::string filename, SavedFct f, std::string plane, double zoom, int datapoints,
                                std::vector<double> origin) {
    PlotParameters params;
    if (plane == "xy" || plane == "yx") {
        params.set_plane({"x1", "x2"});
    } else if (plane == "yz" || plane == "zy") {
        params.set_plane({"x2", "x3"});
    } else if (plane == "zx" || plane == "xz") {
        params.set_plane({"x1", "x3"});
    } else {
        std::cout << "Plane to plot not recognized.";
    }
    params.set_zoom(zoom);
    params.set_origin(origin);

    params.set_npoints(datapoints);
    std::vector<Function<double, 3>> MRAf = {loadfct(f)};
    plot_plane<3>(*world, MRAf, filename, params);
}

void MadnessProcess::cube_plot(std::string filename, SavedFct f, MadMolecule molecule, double zoom, int datapoints,
                               std::vector<double> origin) {
    PlotParameters params;
    std::string outfile = filename + ".cube";

    params.set_zoom(zoom);
    params.set_origin(origin);
    params.set_npoints(datapoints);

    plot_cubefile<3>(*world, loadfct(f), outfile, molecule.mol.cubefile_header(params.origin<3>()), params.npoints(),
                     params.zoom(), params.origin<3>());
}
