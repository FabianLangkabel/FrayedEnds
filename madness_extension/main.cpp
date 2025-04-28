#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "optimization.hpp"
#include "pno_interface.hpp"

namespace nb = nanobind;


NB_MODULE(MadPy, m) {
    nb::class_<real_function_3d>(m,"real_function_3d")
        .def(nb::init<>());

    nb::class_<SavedFct>(m, "SavedFct")
        .def(nb::init<const Function<double,3> &>())
        .def_rw("info", &SavedFct::info);

    nb::class_<Optimization>(m, "Optimization")
        .def(nb::init<const double &, const int &, const double &>())
        .def("plot", &Optimization::plot)
        .def("loadfct", &Optimization::loadfct)
        .def("loadfct_from_file", &Optimization::loadfct_from_file)
        .def("GiveInitialOrbitals", &Optimization::GiveInitialOrbitals)
        .def("GiveRDMsAndRotateOrbitals", &Optimization::GiveRDMsAndRotateOrbitals)
        .def("CreateNuclearPotentialAndRepulsion", &Optimization::CreateNuclearPotentialAndRepulsion)
        .def("ReadInitialOrbitals", &Optimization::ReadInitialOrbitals)
        .def("ReadRDMFilesAndRotateOrbitals", &Optimization::ReadRDMFilesAndRotateOrbitals)
        .def("TransformMatrix", &Optimization::TransformMatrix)
        .def("TransformTensor", &Optimization::TransformTensor)
        .def("CalculateAllIntegrals", &Optimization::CalculateAllIntegrals)
        .def("CalculateCoreEnergy", &Optimization::CalculateCoreEnergy)
        .def("CalculateEnergies", &Optimization::CalculateEnergies)
        .def("CalculateLagrangeMultiplier", &Optimization::CalculateLagrangeMultiplier)
        .def("CalculateLagrangeMultiplierElement_AS_AS", &Optimization::CalculateLagrangeMultiplierElement_AS_AS)
        .def("CalculateLagrangeMultiplierElement_AS_Core", &Optimization::CalculateLagrangeMultiplierElement_AS_Core)
        .def("OptimizeOrbitals", &Optimization::OptimizeOrbitals)
        .def("GetAllActiveOrbitalUpdates", &Optimization::GetAllActiveOrbitalUpdates)
        .def("RotateOrbitalsBackAndUpdateIntegrals", &Optimization::RotateOrbitalsBackAndUpdateIntegrals)
        .def("SaveOrbitals", &Optimization::SaveOrbitals)
        .def("SaveEffectiveHamiltonian", &Optimization::SaveEffectiveHamiltonian)
        .def("GetOrbitals", &Optimization::GetOrbitals)
        .def("GetC", &Optimization::GetC)
        .def("GetHTensor", &Optimization::GetHTensor)
        .def("GetGTensor", &Optimization::GetGTensor)
        .def_rw("nocc", &Optimization::nocc)
        .def_rw("truncation_tol", &Optimization::truncation_tol)
        .def_rw("coulomb_lo", &Optimization::coulomb_lo)
        .def_rw("coulomb_eps", &Optimization::coulomb_eps)
        .def_rw("BSH_lo", &Optimization::BSH_lo)
        .def_rw("BSH_eps", &Optimization::BSH_eps);
    
    nb::class_<PNOInterface>(m, "PNOInterface")
        .def(nb::init<const int &, const std::string &, const double &, const int &, const double &>())
        .def("run", &PNOInterface::run);
}