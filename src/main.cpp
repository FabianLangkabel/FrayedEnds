#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "optimization.hpp"
#include "pno_interface.hpp"
#include "sum_of_gaussians.hpp"
#include "pyfuncfactory.hpp"
#include "integrals.hpp"
#include "eigensolver.hpp"
#include "nwchem_converter.hpp"
#include "minbas.hpp"
#include "madness_process.hpp"
#include "madmolecule.hpp"

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

    nb::class_<madness::real_function_3d>(m, "real_function_3d").def(nb::init<>());

    nb::class_<MadMolecule>(m, "MadMolecule")
        .def(nb::init<>())
        .def("add_atom", &MadMolecule::add_atom)
        .def("to_json", &MadMolecule::to_json)
        .def("compute_nuclear_derivative", &MadMolecule::compute_nuclear_derivative)
        .def("compute_second_nuclear_derivative", &MadMolecule::compute_second_nuclear_derivative)
        .def("nuclear_repulsion_derivative", &MadMolecule::nuclear_repulsion_derivative)
        .def("nuclear_repulsion_second_derivative", &MadMolecule::nuclear_repulsion_second_derivative)
        .def("get_vnuc", &MadMolecule::get_vnuc);

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
        .def("give_initial_orbitals", &Optimization::give_initial_orbitals)
        .def("give_rdm_and_rotate_orbitals", &Optimization::give_rdm_and_rotate_orbitals)
        .def("give_potential_and_repulsion", &Optimization::give_potential_and_repulsion)
        .def("read_initial_orbitals", &Optimization::read_initial_orbitals)
        .def("read_rdm_files_and_rotate_orbitals", &Optimization::read_rdm_files_and_rotate_orbitals)
        .def("calculate_all_integrals", &Optimization::calculate_all_integrals)
        .def("calculate_core_energy", &Optimization::calculate_core_energy)
        .def("calculate_energies", &Optimization::calculate_energies)
        .def("calculate_lagrange_multiplier", &Optimization::calculate_lagrange_multiplier)
        .def("calculate_lagrange_multiplier_element_as_as", &Optimization::calculate_lagrange_multiplier_element_as_as)
        .def("calculate_lagrange_multiplier_element_as_core", &Optimization::calculate_lagrange_multiplier_element_as_core)
        .def("optimize_orbitals", &Optimization::optimize_orbitals)
        .def("get_all_active_orbital_updates", &Optimization::get_all_active_orbital_updates)
        .def("rotate_orbitals_back", &Optimization::rotate_orbitals_back)
        .def("save_orbitals", &Optimization::save_orbitals)
        .def("save_effective_hamiltonian", &Optimization::save_effective_hamiltonian)
        .def("get_orbitals", &Optimization::get_orbitals)
        .def("get_c", &Optimization::get_c)
        .def("get_h_tensor", &Optimization::get_h_tensor)
        .def("get_g_tensor", &Optimization::get_g_tensor)
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
        .def("get_pnos", &PNOInterface::get_pnos)
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
        .def("create_potential", &CoulombPotentialFromChargeDensity::create_potential)
        .def("create_charge_density", &CoulombPotentialFromChargeDensity::create_charge_density);

    nb::class_<RedirectOutput>(m, "RedirectOutput").def(nb::init<const std::string&>());

    nb::class_<PyFuncFactory>(m, "PyFuncFactory")
        .def(nb::init<MadnessProcess&, std::function<double(double, double, double)>&>())
        .def("get_mra_function", &PyFuncFactory::get_mra_function);

    nb::class_<Eigensolver3D>(m, "Eigensolver")
        .def(nb::init<MadnessProcess&>())
        .def("solve", &Eigensolver3D::solve, nb::arg("input_V"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("solve_with_guesses", &Eigensolver3D::solve_with_input_guesses, nb::arg("input_V"),
             nb::arg("input_guesses"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("get_orbitals", &Eigensolver3D::get_orbitals);

    nb::class_<NWChem_Converter>(m, "NWChem_Converter")
        .def(nb::init<MadnessProcess&>())
        .def("read_nwchem_file", &NWChem_Converter::read_nwchem_file)
        .def("get_normalized_aos", &NWChem_Converter::get_normalized_aos)
        .def("get_mos", &NWChem_Converter::get_mos)
        .def("get_vnuc", &NWChem_Converter::get_vnuc)
        .def("get_nuclear_repulsion_energy", &NWChem_Converter::get_nuclear_repulsion_energy);
}