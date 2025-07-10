#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "optimization.hpp"
#include "pno_interface.hpp"
#include "sum_of_gaussians.hpp"
#include "PyFuncFactory.hpp"
#include "integrals.hpp"
#include "eigensolver.hpp"
#include "nwchem_converter.hpp"
#include "molecule.hpp"
#include "plot.hpp"

namespace nb = nanobind;


NB_MODULE(_madpy_impl, m) {
    nb::class_<real_function_3d>(m,"real_function_3d")
        .def(nb::init<>());

    nb::class_<molecule>(m,"molecule")
        .def(nb::init<>())
        .def("add_atom", &molecule::add_atom)
        .def("to_json", &molecule::to_json);

    nb::class_<SavedFct>(m, "SavedFct")
        .def(nb::init<const Function<double,3> &>())
        .def_rw("info", &SavedFct::info)
        .def_rw("type", &SavedFct::type);

    nb::class_<MadnessProcess>(m, "MadnessProcess")
        .def(nb::init<const double &, const long &, const double &, const int &, const int &, const bool &, const int &>())
        .def("loadfct", &MadnessProcess::loadfct)
        .def("plot", &MadnessProcess::plot)
        .def("plane_plot", &MadnessProcess::plane_plot);


    nb::class_<Integrals>(m, "Integrals")
        .def(nb::init<const double &, const int &, const double &, const int &, const int &, const bool &, const int &>())
        .def("hello", &Integrals::hello)
        .def("compute_overlap_integrals", &Integrals::compute_overlap_integrals, nb::arg("all_orbs"))
        .def("compute_potential_integrals", &Integrals::compute_potential_integrals, nb::arg("all_orbs"), nb::arg("potential"))
        .def("compute_kinetic_integrals", &Integrals::compute_kinetic_integrals, nb::arg("all_orbs"))
        .def("compute_two_body_integrals", &Integrals::compute_two_body_integrals, nb::arg("all_orbs"))
        .def("orthonormalize", &Integrals::orthonormalize, nb::arg("all_orbs"), nb::arg("method")="symmetric", nb::arg("rr_thresh")=0.0);

    nb::class_<Optimization>(m, "Optimization")
        .def(nb::init<const double &, const int &, const double &, const int &, const int &, const bool &, const int &>())
        .def("loadfct", &Optimization::loadfct)
        .def("loadfct_from_file", &Optimization::loadfct_from_file)
        .def("GiveInitialOrbitals", &Optimization::GiveInitialOrbitals)
        .def("GiveRDMsAndRotateOrbitals", &Optimization::GiveRDMsAndRotateOrbitals)
        .def("GivePotentialAndRepulsion", &Optimization::GivePotentialAndRepulsion)
        .def("ReadInitialOrbitals", &Optimization::ReadInitialOrbitals)
        .def("ReadRDMFilesAndRotateOrbitals", &Optimization::ReadRDMFilesAndRotateOrbitals)
        .def("CalculateAllIntegrals", &Optimization::CalculateAllIntegrals)
        .def("CalculateCoreEnergy", &Optimization::CalculateCoreEnergy)
        .def("CalculateEnergies", &Optimization::CalculateEnergies)
        .def("CalculateLagrangeMultiplier", &Optimization::CalculateLagrangeMultiplier)
        .def("CalculateLagrangeMultiplierElement_AS_AS", &Optimization::CalculateLagrangeMultiplierElement_AS_AS)
        .def("CalculateLagrangeMultiplierElement_AS_Core", &Optimization::CalculateLagrangeMultiplierElement_AS_Core)
        .def("OptimizeOrbitals", &Optimization::OptimizeOrbitals)
        .def("GetAllActiveOrbitalUpdates", &Optimization::GetAllActiveOrbitalUpdates)
        .def("RotateOrbitalsBack", &Optimization::RotateOrbitalsBack)
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
        .def(nb::init<const std::string &, const double &, const int &, const double &, const int &, const int &, const bool &, const int &>())
        .def("get_nuclear_potential", &PNOInterface::get_nuclear_potential)
        .def("DeterminePNOsAndIntegrals", &PNOInterface::run)
        .def("GetPNOs", &PNOInterface::GetPNOs)
        .def("GetNuclearRepulsion", &PNOInterface::GetNuclearRepulsion);

    nb::class_<CoulombPotentialFromChargeDensity>(m, "CoulombPotentialFromChargeDensity")
        .def(nb::init<const std::vector<double> &, const double &, const std::vector<std::vector<double> > &, const double &, const long &, const double &, const int &, const int &, const bool &, const int &>())
        .def("plot", &CoulombPotentialFromChargeDensity::plot, nb::arg("filename"), nb::arg("f"), nb::arg("axis") = 2, nb::arg("datapoints") = 2001)
        .def("CreatePotential", &CoulombPotentialFromChargeDensity::CreatePotential)
        .def("CreateChargeDens", &CoulombPotentialFromChargeDensity::CreateChargeDens);

    nb::class_<RedirectOutput>(m, "RedirectOutput")
        .def(nb::init<const std::string &>());
    
    nb::class_<PyFuncFactory>(m, "PyFuncFactory")
        .def(nb::init<std::function<double(double, double, double)> &, const double &, const long &, const double &, const int &, const int &, const bool &>())
        .def("GetMRAFunction", &PyFuncFactory::GetMRAFunction);

    nb::class_<Eigensolver3D>(m, "Eigensolver")
        .def(nb::init<const double &, const long &, const double &, const int &, const int &, const bool &, const int &>())
        .def("solve", &Eigensolver3D::solve, nb::arg("input_V"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("solve_with_guesses", &Eigensolver3D::solve_with_input_guesses, nb::arg("input_V"), nb::arg("input_guesses"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("GetOrbitals", &Eigensolver3D::GetOrbitals);

    nb::class_<NWChem_Converter>(m, "NWChem_Converter")
        .def(nb::init<const double &, const int &, const double &>())
        .def("Read_NWChem_File", &NWChem_Converter::read_nwchem_file)
        .def("GetNormalizedAOs", &NWChem_Converter::GetNormalizedAOs)
        .def("GetMOs", &NWChem_Converter::GetMOs);

    nb::class_<Plot>(m, "Plot")
        .def(nb::init<const double &, const int &, const double &>())
        .def("plot", &Plot::plot, nb::arg("filename"), nb::arg("f"), nb::arg("axis") = 2, nb::arg("datapoints") = 2001)
        .def("plane_plot", &Plot::plane_plot, nb::arg("filename"), nb::arg("f"), nb::arg("plane") = "yz", nb::arg("zoom") = 1.0, nb::arg("datapoints") = 151, nb::arg("origin") = std::vector<double>({0.0, 0.0, 0.0}))
        .def("cube_plot", &Plot::cube_plot, nb::arg("filename"), nb::arg("f"), nb::arg("molecule"), nb::arg("zoom") = 1.0, nb::arg("datapoints") = 151, nb::arg("origin") = std::vector<double>({0.0, 0.0, 0.0}));

}