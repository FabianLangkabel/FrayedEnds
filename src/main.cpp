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
#include "minbas.hpp"
#include "MadnessProcess.hpp"

namespace nb = nanobind;

NB_MODULE(_madpy_impl, m) {
    nb::class_<MadnessProcess>(m, "MadnessProcess")
        .def(nb::init<const double&, const int&, const double&, const int&, const int&, const bool&, const int&>(),
             nb::arg("L"), nb::arg("k"), nb::arg("thresh"), nb::arg("initial_level"), nb::arg("truncate_mode"),
             nb::arg("refine"), nb::arg("n_threads"))
        .def("change_nthreads", &MadnessProcess::change_nthreads, nb::arg("n_threads"))
        .def("loadfct", &MadnessProcess::loadfct)
        .def("loadfct_from_file", &MadnessProcess::loadfct_from_file)
        .def("plot", &MadnessProcess::plot)
        .def("plane_plot", &MadnessProcess::plane_plot)
        .def("cube_plot", &MadnessProcess::cube_plot)
        .def_ro("L", &MadnessProcess::L)
        .def_ro("k", &MadnessProcess::k)
        .def_ro("thresh", &MadnessProcess::thresh)
        .def_ro("initial_level", &MadnessProcess::initial_level)
        .def_ro("truncate_mode", &MadnessProcess::truncate_mode)
        .def_ro("refine", &MadnessProcess::refine)
        .def_ro("n_threads", &MadnessProcess::n_threads);

    nb::class_<real_function_3d>(m, "real_function_3d").def(nb::init<>());

    nb::class_<molecule>(m, "molecule")
        .def(nb::init<>())
        .def("add_atom", &molecule::add_atom)
        .def("to_json", &molecule::to_json);

    nb::class_<SavedFct>(m, "SavedFct")
        .def(nb::init<const Function<double, 3>&>())
        .def(nb::init<const std::string&>())
        .def_rw("info", &SavedFct::info)
        .def_rw("type", &SavedFct::type)
        .def("save_to_file", &SavedFct::save_to_file, nb::arg("filepath"));

    nb::class_<Integrals>(m, "Integrals")
        .def(nb::init<MadnessProcess&>())
        .def("hello", &Integrals::hello)
        .def("compute_overlap_integrals", &Integrals::compute_overlap_integrals, nb::arg("all_orbs"), nb::arg("other"))
        .def("compute_potential_integrals", &Integrals::compute_potential_integrals, nb::arg("all_orbs"),
             nb::arg("potential"))
        .def("compute_kinetic_integrals", &Integrals::compute_kinetic_integrals, nb::arg("all_orbs"))
        .def("compute_two_body_integrals", &Integrals::compute_two_body_integrals, nb::arg("all_orbs"))
        .def("transform", &Integrals::transform, nb::arg("orbitals"), nb::arg("matrix"))
        .def("project_out", &Integrals::project_out, nb::arg("kernel"), nb::arg("target"))
        .def("orthonormalize", &Integrals::orthonormalize, nb::arg("all_orbs"), nb::arg("method") = "symmetric",
             nb::arg("rr_thresh") = 0.0);

    nb::class_<Optimization>(m, "Optimization")
        .def(nb::init<MadnessProcess&>())
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
        .def(nb::init<MadnessProcess&, const std::string&>())
        .def("get_nuclear_potential", &PNOInterface::get_nuclear_potential)
        .def("run", &PNOInterface::run)
        .def("get_sto3g", &PNOInterface::get_sto3g)
        .def("GetPNOs", &PNOInterface::GetPNOs)
        .def("get_frozen_core_dim", &PNOInterface::get_frozen_core_dim)
        .def("get_nuclear_repulsion", &PNOInterface::get_nuclear_repulsion);

    nb::class_<MinBasProjector>(m, "MinBasProjector")
        .def(nb::init<MadnessProcess&, const std::string&>())
        .def("run", &MinBasProjector::run)
        .def("get_nuclear_potential", &MinBasProjector::get_nuclear_potential)
        .def("get_basis_name", &MinBasProjector::get_basis_name)
        .def("get_atomic_basis", &MinBasProjector::get_atomic_basis)
        .def("get_nuclear_repulsion", &MinBasProjector::get_nuclear_repulsion);

    nb::class_<CoulombPotentialFromChargeDensity>(m, "CoulombPotentialFromChargeDensity")
        .def(nb::init<MadnessProcess&, const std::vector<double>&, const double&,
                      const std::vector<std::vector<double>>&>())
        .def("CreatePotential", &CoulombPotentialFromChargeDensity::CreatePotential)
        .def("CreateChargeDens", &CoulombPotentialFromChargeDensity::CreateChargeDens);

    nb::class_<RedirectOutput>(m, "RedirectOutput").def(nb::init<const std::string&>());

    nb::class_<PyFuncFactory>(m, "PyFuncFactory")
        .def(nb::init<MadnessProcess&, std::function<double(double, double, double)>&>())
        .def("GetMRAFunction", &PyFuncFactory::GetMRAFunction);

    nb::class_<Eigensolver3D>(m, "Eigensolver")
        .def(nb::init<MadnessProcess&>())
        .def("solve", &Eigensolver3D::solve, nb::arg("input_V"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("solve_with_guesses", &Eigensolver3D::solve_with_input_guesses, nb::arg("input_V"),
             nb::arg("input_guesses"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("GetOrbitals", &Eigensolver3D::GetOrbitals);

    nb::class_<NWChem_Converter>(m, "NWChem_Converter")
        .def(nb::init<MadnessProcess&>())
        .def("Read_NWChem_File", &NWChem_Converter::read_nwchem_file)
        .def("GetNormalizedAOs", &NWChem_Converter::GetNormalizedAOs)
        .def("GetMOs", &NWChem_Converter::GetMOs)
        .def("GetVNuc", &NWChem_Converter::GetVNuc)
        .def("GetNuclearRepulsionEnergy", &NWChem_Converter::GetNuclearRepulsionEnergy);
}