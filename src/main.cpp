#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "optimization.hpp"
#include "pno_interface.hpp"
#include "sum_of_gaussians.hpp"
#include "pyfuncfactory.hpp"
#include "integrals.hpp"
#include "eigensolver.hpp"
#include "nwchem_converter.hpp"
#include "atombas.hpp"
#include "madness_process.hpp"
#include "moleculargeometry.hpp"
#include "open_shell/nwchem_converter_open_shell.hpp"
#include "open_shell/optimization_open_shell.hpp"
#include "open_shell/integrals_open_shell.hpp"
#include "nbtest.hpp"

namespace nb = nanobind;

NB_MODULE(_frayedends_impl, m) {

    nb::class_<MadnessProcess<3>>(m, "MadnessProcess3D")
        .def(nb::init<const double&, const int&, const double&, const int&, const int&, const bool&, const int&>(),
             nb::arg("L"), nb::arg("k"), nb::arg("thresh"), nb::arg("initial_level"), nb::arg("truncate_mode"),
             nb::arg("refine"), nb::arg("n_threads"))
        .def("get_function_defaults", [](MadnessProcess<3>& self) {
            auto t = self.get_function_defaults();
            nb::dict d;
            d["cell_width"] = std::get<0>(t);
            d["k"] = std::get<1>(t);
            d["thresh"] = std::get<2>(t);
            d["initial_level"] = std::get<3>(t);
            d["truncate_mode"] = std::get<4>(t);
            d["refine"] = std::get<5>(t);
            d["n_threads"] = std::get<6>(t);
            return d;
        })
        .def("update_function_defaults", &MadnessProcess<3>::update_function_defaults)
        .def("change_nthreads", &MadnessProcess<3>::change_nthreads, nb::arg("n_threads"))
        .def("loadfct", &MadnessProcess<3>::loadfct)
        .def("loadfct_from_m_file", &MadnessProcess<3>::loadfct_from_m_file)
        .def("save_to_m_file", &MadnessProcess<3>::save_to_m_file)
        .def("load_savedfct_from_m_file", &MadnessProcess<3>::load_savedfct_from_m_file)
        .def("plot", &MadnessProcess<3>::plot)
        .def("plane_plot", &MadnessProcess<3>::plane_plot)
        .def("cube_plot", &MadnessProcess<3>::cube_plot)
        .def_rw("L", &MadnessProcess<3>::L)
        .def_rw("k", &MadnessProcess<3>::k)
        .def_rw("thresh", &MadnessProcess<3>::thresh)
        .def_rw("initial_level", &MadnessProcess<3>::initial_level)
        .def_rw("truncate_mode", &MadnessProcess<3>::truncate_mode)
        .def_rw("refine", &MadnessProcess<3>::refine)
        .def("evaluate", &MadnessProcess<3>::evaluate)
        .def_rw("n_threads", &MadnessProcess<3>::n_threads);
    
    nb::class_<MadnessProcess<2>>(m, "MadnessProcess2D")
        .def(nb::init<const double&, const int&, const double&, const int&, const int&, const bool&, const int&>(),
             nb::arg("L"), nb::arg("k"), nb::arg("thresh"), nb::arg("initial_level"), nb::arg("truncate_mode"),
             nb::arg("refine"), nb::arg("n_threads"))
        .def("get_function_defaults", [](MadnessProcess<2>& self) {
            auto t = self.get_function_defaults();
            nb::dict d;
            d["cell_width"] = std::get<0>(t);
            d["k"] = std::get<1>(t);
            d["thresh"] = std::get<2>(t);
            d["initial_level"] = std::get<3>(t);
            d["truncate_mode"] = std::get<4>(t);
            d["refine"] = std::get<5>(t);
            d["n_threads"] = std::get<6>(t);
            return d;
        })
        .def("update_function_defaults", &MadnessProcess<2>::update_function_defaults)
        .def("change_nthreads", &MadnessProcess<2>::change_nthreads, nb::arg("n_threads"))
        .def("loadfct", &MadnessProcess<2>::loadfct)
        .def("loadfct_from_m_file", &MadnessProcess<2>::loadfct_from_m_file)
        .def("save_to_m_file", &MadnessProcess<2>::save_to_m_file)
        .def("load_savedfct_from_m_file", &MadnessProcess<2>::load_savedfct_from_m_file)
        .def("plot", &MadnessProcess<2>::plot)
        .def("plane_plot", &MadnessProcess<2>::plane_plot)
        .def_rw("L", &MadnessProcess<2>::L)
        .def_rw("k", &MadnessProcess<2>::k)
        .def_rw("thresh", &MadnessProcess<2>::thresh)
        .def_rw("initial_level", &MadnessProcess<2>::initial_level)
        .def_rw("truncate_mode", &MadnessProcess<2>::truncate_mode)
        .def_rw("refine", &MadnessProcess<2>::refine)
        .def("evaluate", &MadnessProcess<2>::evaluate)
        .def_rw("n_threads", &MadnessProcess<2>::n_threads);

    nb::class_<madness::real_function_3d>(m, "real_function_3d").def(nb::init<>());
    nb::class_<madness::real_function_2d>(m, "real_function_2d").def(nb::init<>());

    nb::class_<MolecularGeometry>(m, "MolecularGeometry")
        .def(nb::init<const std::string&>())
        .def_rw("units", &MolecularGeometry::units)
        .def("add_atom", &MolecularGeometry::add_atom)
        .def("to_json", &MolecularGeometry::to_json)
        .def("get_nuclear_repulsion", &MolecularGeometry::get_nuclear_repulsion)
        .def("get_nuclear_charge", &MolecularGeometry::get_nuclear_charge)
        .def("get_core_n_electrons", &MolecularGeometry::get_core_n_electrons)
        .def("molecular_potential_derivative", &MolecularGeometry::molecular_potential_derivative)
        .def("molecular_potential_second_derivative", &MolecularGeometry::molecular_potential_second_derivative)
        .def("nuclear_repulsion_derivative", &MolecularGeometry::nuclear_repulsion_derivative)
        .def("nuclear_repulsion_second_derivative", &MolecularGeometry::nuclear_repulsion_second_derivative)
        .def("get_vnuc", &MolecularGeometry::get_vnuc);

    nb::class_<SavedFct<3>>(m, "SavedFct3D")
        .def(nb::init<const Function<double, 3>&>())
        .def(nb::init<const std::string&>())
        .def_rw("info", &SavedFct<3>::info)
        .def("save_to_file", &SavedFct<3>::save_to_file, nb::arg("filepath"))
        .def("load_from_file", &SavedFct<3>::load_from_file, nb::arg("filepath"));

    nb::class_<SavedFct<2>>(m, "SavedFct2D")
        .def(nb::init<const Function<double, 2>&>())
        .def(nb::init<const std::string&>())
        .def_rw("info", &SavedFct<2>::info)
        .def("save_to_file", &SavedFct<2>::save_to_file, nb::arg("filepath"))
        .def("load_from_file", &SavedFct<2>::load_from_file, nb::arg("filepath"));

    nb::class_<Integrals<3>>(m, "Integrals3D")
        .def(nb::init<MadnessProcess<3>&>())
        .def("override_numerical_parameters", static_cast<void (Integrals<3>::*)(double, double, double)>(&Integrals<3>::override_numerical_parameters))
        .def("compute_overlap_integrals", &Integrals<3>::nb_compute_overlap_integrals)
        .def("compute_potential_integrals", &Integrals<3>::nb_compute_potential_integrals)
        .def("compute_kinetic_integrals", &Integrals<3>::nb_compute_kinetic_integrals)
        .def("compute_two_body_integrals", &Integrals<3>::nb_compute_two_body_integrals)
        .def("compute_frozen_core_interaction", &Integrals<3>::nb_compute_frozen_core_interaction)
        .def("compute_effective_hamiltonian", &Integrals<3>::nb_compute_effective_hamiltonian)
        .def("compute_overlap_integrals_old", &Integrals<3>::nb_compute_overlap_integrals_old)
        .def("compute_potential_integrals_old", &Integrals<3>::nb_compute_potential_integrals_old)
        .def("compute_kinetic_integrals_old", &Integrals<3>::nb_compute_kinetic_integrals_old)
        .def("compute_two_body_integrals_old", &Integrals<3>::nb_compute_two_body_integrals_old)
        .def("compute_frozen_core_interaction_old", &Integrals<3>::nb_compute_frozen_core_interaction_old)
        .def("transform", &Integrals<3>::transform, nb::arg("orbitals"), nb::arg("matrix"))
        .def("project_out", &Integrals<3>::project_out, nb::arg("kernel"), nb::arg("target"))
        .def("project_on", &Integrals<3>::project_on, nb::arg("kernel"), nb::arg("target"))
        .def("normalize", &Integrals<3>::normalize)
        .def("orthonormalize", &Integrals<3>::orthonormalize);

    nb::class_<Integrals<2>>(m, "Integrals2D")
        .def(nb::init<MadnessProcess<2>&>())
        .def("override_numerical_parameters", static_cast<void (Integrals<2>::*)(double, double, double)>(&Integrals<2>::override_numerical_parameters))
        .def("compute_overlap_integrals", &Integrals<2>::nb_compute_overlap_integrals)
        .def("compute_potential_integrals", &Integrals<2>::nb_compute_potential_integrals)
        .def("compute_kinetic_integrals", &Integrals<2>::nb_compute_kinetic_integrals)
        .def("compute_two_body_integrals", &Integrals<2>::nb_compute_two_body_integrals)
        .def("compute_frozen_core_interaction", &Integrals<2>::nb_compute_frozen_core_interaction)
        .def("compute_effective_hamiltonian", &Integrals<2>::nb_compute_effective_hamiltonian)
        .def("compute_overlap_integrals_old", &Integrals<2>::nb_compute_overlap_integrals_old)
        .def("compute_potential_integrals_old", &Integrals<2>::nb_compute_potential_integrals_old)
        .def("compute_kinetic_integrals_old", &Integrals<2>::nb_compute_kinetic_integrals_old)
        .def("compute_two_body_integrals_old", &Integrals<2>::nb_compute_two_body_integrals_old)
        .def("compute_frozen_core_interaction_old", &Integrals<2>::nb_compute_frozen_core_interaction_old)
        .def("transform", &Integrals<2>::transform, nb::arg("orbitals"), nb::arg("matrix"))
        .def("project_out", &Integrals<2>::project_out, nb::arg("kernel"), nb::arg("target"))
        .def("project_on", &Integrals<2>::project_on, nb::arg("kernel"), nb::arg("target"))
        .def("normalize", &Integrals<2>::normalize)
        .def("orthonormalize", &Integrals<2>::orthonormalize);

    nb::class_<Integrals_open_shell<3>>(m, "Integrals_open_shell_3D")
        .def(nb::init<MadnessProcess<3>&>())
        .def("compute_potential_integrals", &Integrals_open_shell<3>::nb_compute_potential_integrals, nb::arg("alpha_orbitals"), nb::arg("beta_orbitals"), nb::arg("potential"))
        .def("compute_kinetic_integrals", &Integrals_open_shell<3>::nb_compute_kinetic_integrals, nb::arg("alpha_orbitals"), nb::arg("beta_orbitals"))
        .def("compute_two_body_integrals", &Integrals_open_shell<3>::nb_compute_two_body_integrals, nb::arg("alpha_orbitals"), nb::arg("beta_orbitals"))
        .def("compute_effective_hamiltonian", &Integrals_open_shell<3>::nb_compute_effective_hamiltonian, nb::arg("core_alpha_orbitals"), nb::arg("core_beta_orbitals"), nb::arg("active_alpha_orbitals"), nb::arg("active_beta_orbitals"), nb::arg("potential"), nb::arg("energy_offset"))
        .def("override_numerical_parameters", static_cast<void (Integrals_open_shell<3>::*)(double, double, double, double, double)>(&Integrals_open_shell<3>::override_numerical_parameters), nb::arg("truncation_tol"), nb::arg("coulomb_lo"), nb::arg("coulomb_eps"), nb::arg("BSH_lo"), nb::arg("BSH_eps"));
        

    nb::class_<Optimization<3>>(m, "Optimization3D")
        .def(nb::init<MadnessProcess<3>&>())
        .def("override_numerical_parameters", &Optimization<3>::override_numerical_parameters)
        .def("get_numerical_parameters", &Optimization<3>::get_numerical_parameters)
        .def("give_initial_orbitals", &Optimization<3>::give_initial_orbitals)
        .def("give_rdm_and_rotate_orbitals", &Optimization<3>::give_rdm_and_rotate_orbitals)
        .def("give_potential_and_repulsion", &Optimization<3>::give_potential_and_repulsion)
        .def("calculate_all_integrals", &Optimization<3>::calculate_all_integrals)
        .def("calculate_energies", &Optimization<3>::calculate_energies)
        .def("calculate_lagrange_multiplier", &Optimization<3>::calculate_lagrange_multiplier)
        .def("calculate_lagrange_multiplier_element_as_as",
             &Optimization<3>::calculate_lagrange_multiplier_element_as_as)
        .def("calculate_lagrange_multiplier_element_as_core",
             &Optimization<3>::calculate_lagrange_multiplier_element_as_core)
        .def("optimize_orbitals", &Optimization<3>::optimize_orbitals)
        .def("get_all_active_orbital_updates", &Optimization<3>::get_all_active_orbital_updates)
        .def("rotate_orbitals_back", &Optimization<3>::rotate_orbitals_back)
        .def("set_orthonormalization_method", &Optimization<3>::set_orthonormalization_method)
        .def("get_orbitals", &Optimization<3>::get_orbitals)
        .def("get_effective_hamiltonian", &Optimization<3>::get_effective_hamiltonian)
        .def("calculate_all_integrals_old", &Optimization<3>::calculate_all_integrals_old)
        .def("calculate_energies_old", &Optimization<3>::calculate_energies_old)
        .def("calculate_lagrange_multiplier_old", &Optimization<3>::calculate_lagrange_multiplier_old)
        .def("calculate_lagrange_multiplier_element_as_as_old",
             &Optimization<3>::calculate_lagrange_multiplier_element_as_as_old)
        .def("calculate_lagrange_multiplier_element_as_core_old",
             &Optimization<3>::calculate_lagrange_multiplier_element_as_core_old)
        .def("optimize_orbitals_old", &Optimization<3>::optimize_orbitals_old)
        .def("get_all_active_orbital_updates_old", &Optimization<3>::get_all_active_orbital_updates_old)
        .def("calculate_core_energy_old", &Optimization<3>::calculate_core_energy_old)
        .def("get_c", &Optimization<3>::get_c)
        .def("get_h_tensor", &Optimization<3>::get_h_tensor)
        .def("get_g_tensor", &Optimization<3>::get_g_tensor);

    nb::class_<Optimization<2>>(m, "Optimization2D")
        .def(nb::init<MadnessProcess<2>&>())
        .def("give_initial_orbitals", &Optimization<2>::give_initial_orbitals)
        .def("give_rdm_and_rotate_orbitals", &Optimization<2>::give_rdm_and_rotate_orbitals)
        .def("give_potential_and_repulsion", &Optimization<2>::give_potential_and_repulsion)
        .def("calculate_all_integrals", &Optimization<2>::calculate_all_integrals)
        .def("calculate_energies", &Optimization<2>::calculate_energies)
        .def("calculate_lagrange_multiplier", &Optimization<2>::calculate_lagrange_multiplier)
        .def("calculate_lagrange_multiplier_element_as_as",
             &Optimization<2>::calculate_lagrange_multiplier_element_as_as)
        .def("calculate_lagrange_multiplier_element_as_core",
             &Optimization<2>::calculate_lagrange_multiplier_element_as_core)
        .def("optimize_orbitals", &Optimization<2>::optimize_orbitals)
        .def("get_all_active_orbital_updates", &Optimization<2>::get_all_active_orbital_updates)
        .def("rotate_orbitals_back", &Optimization<2>::rotate_orbitals_back)
        .def("set_orthonormalization_method", &Optimization<2>::set_orthonormalization_method)
        .def("get_orbitals", &Optimization<2>::get_orbitals)
        .def("calculate_core_energy_old", &Optimization<2>::calculate_core_energy_old)
        .def("get_c", &Optimization<2>::get_c)
        .def("get_h_tensor", &Optimization<2>::get_h_tensor)
        .def("get_g_tensor", &Optimization<2>::get_g_tensor);

    nb::class_<Optimization_open_shell<3>>(m, "Optimization_open_shell_3D")
        .def(nb::init<MadnessProcess<3>&>())
        .def("give_initial_orbitals", &Optimization_open_shell<3>::give_initial_orbitals)
        .def("give_rdm_and_rotate_orbitals", &Optimization_open_shell<3>::give_rdm_and_rotate_orbitals)
        .def("give_potential_and_repulsion", &Optimization_open_shell<3>::give_potential_and_repulsion)
        .def("calculate_all_integrals", &Optimization_open_shell<3>::calculate_all_integrals)
        .def("calculate_energies", &Optimization_open_shell<3>::calculate_energies)
        .def("calculate_lagrange_multiplier", &Optimization_open_shell<3>::calculate_lagrange_multiplier)
        .def("calculate_lagrange_multiplier_element_as_as", &Optimization_open_shell<3>::calculate_lagrange_multiplier_element_as_as)
        .def("calculate_lagrange_multiplier_element_as_core", &Optimization_open_shell<3>::calculate_lagrange_multiplier_element_as_core)
        .def("optimize_orbitals", &Optimization_open_shell<3>::optimize_orbitals)
        .def("get_all_active_orbital_updates", &Optimization_open_shell<3>::get_all_active_orbital_updates)
        .def("rotate_orbitals_back", &Optimization_open_shell<3>::rotate_orbitals_back)
        .def("get_effective_hamiltonian", &Optimization_open_shell<3>::get_effective_hamiltonian)
        .def("get_orbitals", &Optimization_open_shell<3>::get_orbitals)
        .def("override_numerical_parameters", &Optimization_open_shell<3>::override_numerical_parameters, nb::arg("truncation_tol"), nb::arg("coulomb_lo"), nb::arg("coulomb_eps"), nb::arg("BSH_lo"), nb::arg("BSH_eps"));

    nb::class_<PNOInterface>(m, "PNOInterface")
        .def(nb::init<MadnessProcess<3>&, const std::string&>())
        .def("get_nuclear_potential", &PNOInterface::get_nuclear_potential)
        .def("run", &PNOInterface::run)
        .def("get_sto3g", &PNOInterface::get_sto3g)
        .def("get_pnos", &PNOInterface::get_pnos)
        .def("get_frozen_core_dim", &PNOInterface::get_frozen_core_dim)
        .def("get_nuclear_repulsion", &PNOInterface::get_nuclear_repulsion);

    nb::class_<AtomBasProjector>(m, "AtomBasProjector")
        .def(nb::init<MadnessProcess<3>&, const std::string&, const std::string&>())
        .def("run", &AtomBasProjector::run)
        .def("get_nuclear_potential", &AtomBasProjector::get_nuclear_potential)
        .def("get_basis_name", &AtomBasProjector::get_basis_name)
        .def("get_atomic_basis", &AtomBasProjector::get_atomic_basis)
        .def("solve_scf", &AtomBasProjector::solve_scf)
        .def("get_nuclear_repulsion", &AtomBasProjector::get_nuclear_repulsion);

    nb::class_<CoulombPotentialFromChargeDensity>(m, "CoulombPotentialFromChargeDensity")
        .def(nb::init<MadnessProcess<3>&, const std::vector<double>&, const double&,
                      const std::vector<std::vector<double>>&>())
        .def("create_potential", &CoulombPotentialFromChargeDensity::create_potential)
        .def("create_charge_density", &CoulombPotentialFromChargeDensity::create_charge_density);

    nb::class_<RedirectOutput>(m, "RedirectOutput").def(nb::init<const std::string&>());

    nb::class_<PyFuncFactory<3>>(m, "PyFuncFactory3D")
        .def(nb::init<MadnessProcess<3>&, std::function<double(double, double, double)>&>())
        .def("get_mra_function", &PyFuncFactory<3>::get_mra_function);
    
    nb::class_<PyFuncFactory<2>>(m, "PyFuncFactory2D")
        .def(nb::init<MadnessProcess<2>&, std::function<double(double, double)>&>())
        .def("get_mra_function", &PyFuncFactory<2>::get_mra_function);

    nb::class_<Eigensolver<3>>(m, "Eigensolver3D")
        .def(nb::init<MadnessProcess<3>&>())
        .def("solve", &Eigensolver<3>::solve, nb::arg("input_V"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("solve_with_guesses", &Eigensolver<3>::solve_with_input_guesses, nb::arg("input_V"),
             nb::arg("input_guesses"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("get_orbitals", &Eigensolver<3>::get_orbitals);

    nb::class_<Eigensolver<2>>(m, "Eigensolver2D")
        .def(nb::init<MadnessProcess<2>&>())
        .def("solve", &Eigensolver<2>::solve, nb::arg("input_V"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("solve_with_guesses", &Eigensolver<2>::solve_with_input_guesses, nb::arg("input_V"),
             nb::arg("input_guesses"), nb::arg("num_levels"), nb::arg("max_iter"))
        .def("get_orbitals", &Eigensolver<2>::get_orbitals);

    nb::class_<NWChem_Converter>(m, "NWChem_Converter")
        .def(nb::init<MadnessProcess<3>&>())
        .def("read_nwchem_file", &NWChem_Converter::read_nwchem_file)
        .def("get_normalized_aos", &NWChem_Converter::get_normalized_aos)
        .def("get_mos", &NWChem_Converter::get_mos)
        .def("get_vnuc", &NWChem_Converter::get_vnuc)
        .def("get_nuclear_repulsion_energy", &NWChem_Converter::get_nuclear_repulsion_energy);

    nb::class_<NWChem_Converter_open_shell>(m, "NWChem_Converter_open_shell")
        .def(nb::init<MadnessProcess<3>&>())
        .def("read_nwchem_file", &NWChem_Converter_open_shell::read_nwchem_file)
        .def("get_normalized_aos", &NWChem_Converter_open_shell::get_normalized_aos)
        .def("get_alpha_mos", &NWChem_Converter_open_shell::get_alpha_mos)
        .def("get_beta_mos", &NWChem_Converter_open_shell::get_beta_mos)
        .def("get_vnuc", &NWChem_Converter_open_shell::get_vnuc)
        .def("get_nuclear_repulsion_energy", &NWChem_Converter_open_shell::get_nuclear_repulsion_energy);

    nb::class_<NBArraytest>(m, "NBArraytest")
        .def(nb::init<const std::tuple<unsigned int, unsigned int>&>())
        .def("fill_array", &NBArraytest::fill_array)
        .def("to_numpy", &NBArraytest::to_numpy)
        .def("double_all", &NBArraytest::double_all)
        .def("get_unsafe", &NBArraytest::get_unsafe)
        .def("get_cast", &NBArraytest::get_cast)
        .def("get_capsule_simple", &NBArraytest::get_capsule_simpl)
        .def("get_capsule_convoluded", &NBArraytest::get_capsule_convoluded)
        .def("explode", &NBArraytest::explode);
}