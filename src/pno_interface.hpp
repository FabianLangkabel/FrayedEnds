#pragma once

#include "functionsaver.hpp"
#include <iomanip>
#include <madness/mra/vmra.h>
#include <madness/chem/SCF.h>
#include <madness/chem/nemo.h>
#include <madness/chem/TDHF.h>
#include <madness/chem/PNO.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <fstream>
#include <regex>
#include <utility> // For std::pair
#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "integrals.hpp"
#include "madness_process.hpp"

using namespace madness;
namespace nb = nanobind;

// DEFINE PARAMETER TAGS FOR THE INPUT FILE
const std::string TAG_PNO = "pno";
const std::string TAG_F12 = "f12";
const std::string TAG_CP = "computeprotocol";

template <typename T1, typename T2> std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& v) {
    os << "(" << v.first << "," << v.second << ")";
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (auto i = 0; i < v.size(); ++i) {
        os << v[i] << " ";
    }
    os << "]";
    return os;
}

inline std::pair<size_t, char**> stringToCharPointerArray(const std::string& input) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;
    std::string current_token;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (c == '"' && (i == 0 || input[i - 1] != '\\')) { // Handle quotes
            in_quotes = !in_quotes;
            if (!in_quotes) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else if (std::isspace(c) && !in_quotes) { // Handle spaces outside quotes
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c; // Add character to the current token
        }
    }

    // Add the last token if it exists
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    // Allocate memory for the array of char* pointers
    char** charArray = new char*[tokens.size()];

    // Copy each token into the array
    for (size_t i = 0; i < tokens.size(); ++i) {
        charArray[i] = new char[tokens[i].size() + 1]; // +1 for null terminator
        std::strcpy(charArray[i], tokens[i].c_str());
    }

    return {tokens.size(), charArray};
}

// Clean up memory
inline void freeCharPointerArray(char** charArray, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        delete[] charArray[i]; // Free each string
    }
    delete[] charArray; // Free the array of pointers
}

class PNOInterface {
  public:
    PNOInterface(MadnessProcess<3>& mp, std::string argv) : madness_process(mp) {

        auto [argc, charArray] = stringToCharPointerArray(argv);
        parser = commandlineparser(argc, charArray);
        freeCharPointerArray(charArray, argc);
    }
    ~PNOInterface() {
        basis.clear();
        Vnuc.clear();
    }

    SavedFct<3> get_nuclear_potential() { return SavedFct<3>(Vnuc); }

    void run(const size_t basis_size) {
        std::cout.precision(6);
        if (madness_process.world->rank() == 0) {
            std::cout << "\n\n";
            std::cout << "-------------------------------------------------------------------------------------\n";
            std::cout << "SOLVING MRA-PNO-F12 as described in \n";
            std::cout << "J.S. Kottmann, F.A. Bischoff, E.F. Valeev\n";
            std::cout << "Direct determination of optimal pair-natural orbitals in a real-space representation:\n";
            std::cout << "the second-order Møller-Plesset energy\n";
            std::cout << "Journal of Chemical Physics ... 2020\n";
            std::cout << "-------------------------------------------------------------------------------------\n";
            std::cout << "\n\n";

            std::cout << "This script will run PNO-MP2 and print out tensors in binary\n";
        }

        // Compute the SCF Reference
        const double time_scf_start = wall_time();
        auto nemo = std::make_shared<Nemo>(*(madness_process.world), parser);
        nemo->get_calc()->param.print();
        const double scf_energy = nemo->value();
        if (madness_process.world->rank() == 0)
            print("nemo energy: ", scf_energy);
        if (madness_process.world->rank() == 0)
            printf(" at time %.1f\n", wall_time());
        const double time_scf_end = wall_time();
        // assert that no nemo corrfactor is actually used (not yet supported in PNO-MP2)
        if (nemo->ncf->type() != madness::NuclearCorrelationFactor::None) {
            MADNESS_EXCEPTION(
                "Nuclear Correlation Factors not yet supported in MRA-PNOs. Add ncf (none,1.0) to your dft input", 1);
        }
        {
            Vnuc = nemo->ncf->U2();
            nuclear_repulsion = nemo->get_calc()->molecule.nuclear_repulsion_energy();
        }
        if (madness_process.world->rank() == 0) {
            parser.print_map();
        }

        int n_excitations = -1;
        if (parser.key_exists("tdhf")) {
            std::smatch m;
            std::string s = parser.value("tdhf");
            if (std::regex_search(s, m, std::regex(R"(nexcitations\s+(\d+))")))
                n_excitations = std::stoi(m[1]);
        }

        if (madness_process.world->rank() == 0) {
            printf("DEBUG: Found n_excitations = %d\n", n_excitations);
        }

        std::vector<CC_vecfunction> cis_roots;
        if (n_excitations > -1) {
            if (madness_process.world->rank() == 0) {
                std::cout << "--------------------------------------------------\n";
                std::cout << "TDHF CIS will be computed with " << n_excitations << " excitations \n";
                std::cout << "--------------------------------------------------\n";
            }

            // Compute TDHF CIS vectors
            const double time_cis_start = wall_time();;
            TDHF tdhf(*(madness_process.world), parser, nemo);
            if (madness_process.world->rank() == 0) {
                std::cout << "CIS parameters: \n";
            }
            tdhf.get_parameters().print("cis", "end");
            tdhf.prepare_calculation();
            cis_roots = tdhf.solve_cis();
            const double time_cis_end = wall_time();

            size_t freeze_offset = 0;
            PNOParameters temp_params(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
            freeze_offset = temp_params.freeze();

            if (madness_process.world->rank() == 0) {
                std::cout << "Total CIS roots found: " << cis_roots.size() << "\n";
            }

            for (size_t ex = 0; ex < cis_roots.size(); ++ex) {
                const auto& xvec = cis_roots[ex].functions;

                if (madness_process.world->rank() == 0) {
                    std::cout << "Saving CIS x-vectors for excitation " << ex
                              << " (contains " << xvec.size() << " functions)" << std::endl;
                }

                for (auto const& [idx, cc_func] : xvec) {
                    std::string filename = std::to_string(ex) + "_x" + std::to_string(idx);
                    save(cc_func.function, filename);
                    cis_x_functions.push_back(cc_func.function);
                }
            }

            if (madness_process.world->rank() == 0) {
                std::cout << std::setfill(' ');
                std::cout << "\n\n\n";
                std::cout << "--------------------------------------------------\n";
                std::cout << "TDHF CIS ended \n";
                std::cout << "--------------------------------------------------\n";
                std::cout << std::setw(25) << "time cis" << " = " << time_cis_end - time_cis_start << "\n";
                std::cout << "--------------------------------------------------\n";
            }
        } else {
            if (madness_process.world->rank() == 0) {
                    std::cout << "\n\n\n";
                    std::cout << "--------------------------------------------------\n";
                    std::cout << "TDHF CIS will be skipped since no excitations were requested \n";
                    std::cout << "--------------------------------------------------\n";
            }
        }

        // Compute MRA-PNO-MP2-F12
        const double time_pno_start = wall_time();
        PNOParameters parameters(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
        F12Parameters paramf12(*(madness_process.world), parser, parameters, TAG_F12);
        PNO pno(*(madness_process.world), *nemo, parameters, paramf12);

        std::vector<PNOPairs> all_pairs;
        pno.solve(all_pairs);
        const double time_pno_end = wall_time();

        if (madness_process.world->rank() == 0) {
            std::cout << std::setfill(' ');
            std::cout << "\n\n\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "MRA-PNO-MP2 ended \n";
            std::cout << "--------------------------------------------------\n";
            std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
            std::cout << std::setw(25) << "energy scf" << " = " << scf_energy << "\n";
            std::cout << "--------------------------------------------------\n";
        }

        // Compute CISPD-PNO
        const double time_cispd_start = wall_time();
        if (n_excitations > -1) {
            for (size_t ex = 0; ex < cis_roots.size(); ++ex) {
                parser.set_keyval("pno", parser.value("pno") + "; cispd_number " + std::to_string(ex) + "; cispd_energy " + std::to_string(cis_roots[ex].omega));

                PNOParameters cispd_parameters(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
                F12Parameters cispd_paramf12(*(madness_process.world), parser, cispd_parameters, TAG_F12);
                PNO pno_cispd(*(madness_process.world), *nemo, cispd_parameters, cispd_paramf12);

                std::vector<PNOPairs> cispd_pairs;
                pno_cispd.solve(cispd_pairs);

                for (size_t i = 1; i < cispd_pairs.size(); ++i)
                    all_pairs.push_back(std::move(cispd_pairs[i]));
            }
        }
        const double time_cispd_end = wall_time();

        if (madness_process.world->rank() == 0) {
            std::cout << std::setfill(' ');
            std::cout << "\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "CISPD ended \n";
            std::cout << "--------------------------------------------------\n";
            std::cout << std::setw(25) << "time cispd" << " = " << time_cispd_end - time_cispd_start << "\n";
            std::cout << "--------------------------------------------------\n";
        }

        if (madness_process.world->rank() == 0) {
            parser.print_map();
        }

        if (madness_process.world->rank() == 0) {
            std::cout << "restarting PNO to reload all pairs that converged before and were frozen\n";
        }

        const std::string pno_base = parser.value("pno");
        all_pairs.clear();

        pno.param.set_user_defined_value<std::string>("restart", "all");
        pno.param.set_user_defined_value<std::string>("no_opt", "all");
        pno.param.set_user_defined_value<std::string>("no_guess", "all");
        pno.param.set_user_defined_value<std::string>("adaptive_solver", "none");

        pno.solve(all_pairs);

        if (madness_process.world->rank() == 0) {
            parser.print_map();
        }
        if (n_excitations > -1) {
            for (size_t ex = 0; ex < cis_roots.size(); ++ex) {
                parser.set_keyval("pno", pno_base + "; cispd_number " + std::to_string(ex) + "; cispd_energy " + std::to_string(cis_roots[ex].omega));

                PNOParameters cispd_reload_parameters(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
                F12Parameters cispd_reload_paramf12(*(madness_process.world), parser, cispd_reload_parameters, TAG_F12);
                PNO pno_reload(*(madness_process.world), *nemo, cispd_reload_parameters, cispd_reload_paramf12);

                pno_reload.param.set_user_defined_value<std::string>("restart", "all");
                pno_reload.param.set_user_defined_value<std::string>("no_opt",  "mp2");
                pno_reload.param.set_user_defined_value<std::string>("no_guess","all");
                pno_reload.param.set_user_defined_value<std::string>("adaptive_solver","none");
                pno_reload.param.set_user_defined_value<std::string>("no_compute", "mp2");

                std::vector<PNOPairs> cispd_reload_pairs;
                pno_reload.solve(cispd_reload_pairs);

                for (size_t i = 1; i < cispd_reload_pairs.size(); ++i)
                    all_pairs.push_back(std::move(cispd_reload_pairs[i]));
                }
        }

        double mp2_energy = 0.0;
        if (madness_process.world->rank() == 0) {
            std::cout << std::setw(25) << "time pno" << " = " << time_pno_end - time_pno_start << "\n";
            std::cout << std::setw(25) << "time cispd" << " = " << time_cispd_end - time_cispd_start << "\n";
        }

        for (const auto& pairs : all_pairs) {
            if (pairs.type == CISPD_PAIRTYPE) {
                if (madness_process.world->rank() == 0) {
                    const double delta = pairs.energies.total_energy();
                    std::cout << "\n================CIS(D) Excitation "
                              << pairs.cis.number << "================\n";
                    std::cout << std::setw(25) << "omega(CIS)"
                              << " = " << pairs.cis.omega << "\n";
                    std::cout << std::setw(25) << "delta(CIS(D))"
                              << " = " << delta << "\n";
                    std::cout << std::setw(25) << "omega(CIS(D))"
                              << " = " << pairs.cis.omega + delta << "\n";
                    std::cout << "================================================\n";
                }
                continue;
            }
            if (pairs.type == MP2_PAIRTYPE) {
                mp2_energy = pairs.energies.total_energy();
            }

            std::pair<size_t, size_t> ranks = pno.get_average_rank(pairs.pno_ij);
            if (madness_process.world->rank() == 0) {
                std::string name;
                std::stringstream ss;
                ss << pairs.type;
                ss >> name;
                std::cout << std::setw(25) << "energy " + name << " = " << pairs.energies.total_energy() << "\n";
                std::cout << std::setw(25) << "average pno rank " + name << " = " << ranks.first << "\n";
                std::cout << std::setw(25) << "max pno rank " + name << " = " << ranks.second << "\n";
            }
        }
        if (madness_process.world->rank() == 0 and mp2_energy != 0.0) {
            std::cout << "--------------------------------------------------\n";
            std::cout << std::setw(25) << "energy(total)" << " = " << scf_energy + mp2_energy << "\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "\n\n\n";
        }

        // compute orthogonalized mp2 basis and print out hamiltonian tensors
        std::cout << std::setprecision(8);
        std::cout << std::fixed;
        std::cout << std::showpos;

        const double h_thresh = 1.e-7; // neglect integrals
        double thresh = parameters.thresh();

        thresh = std::min(thresh, 1.e-4);
        if (madness_process.world->rank() == 0)
            std::cout << "Tightening thresholds to " << thresh << " for post-processing\n";
        FunctionDefaults<3>::set_thresh(thresh);

        vecfuncT reference = nemo->get_calc()->amo;
        const size_t npno = basis_size - reference.size();

        vecfuncT mp2_pnos, cispd_pnos;
        std::vector<double> mp2_occ, cispd_occ;
        std::vector<std::pair<size_t, size_t>> mp2_ids, cispd_ids;
        std::vector<std::string> mp2_labels, cispd_labels;

        for (auto& pairs : all_pairs) {
            if (pairs.type != MP2_PAIRTYPE && pairs.type != CISPD_PAIRTYPE) {
                continue;
            }
            const auto& pno_ij = pairs.pno_ij;
            const auto& rdm_evals = pairs.rdm_evals_ij;

            const bool only_diag = true;

            std::vector<real_function_3d> all_current_pnos;
            std::vector<double> all_current_occ;
            std::vector<std::pair<size_t, size_t>> all_current_ids;

            // collect PNOs from all pairs and sort by occupation number, keeping pair information via name
            for (ElectronPairIterator it = pno.pit(); it; ++it) {
                if (only_diag and not it.diagonal()) {
                    if (madness_process.world->rank() == 0)
                        std::cout << "skipping pair (not diagonal) " << it.name() << "\n";
                    continue;
                }
                if (it.ij() >= pno_ij.size() || pno_ij[it.ij()].size() == 0) continue;

                if (madness_process.world->rank() == 0)
                    std::cout << "adding " << it.name() << " from " << pairs.type << "\n";

                const auto& pair = pno_ij[it.ij()];

                all_current_pnos.insert(all_current_pnos.end(), pair.begin(), pair.end());

                for (auto ii = 0; ii < rdm_evals[it.ij()].size(); ++ii) {
                    all_current_occ.push_back(rdm_evals[it.ij()][ii]);
                    all_current_ids.push_back(
                        std::make_pair(it.i(), it.j())); // for each eigenvalue ~ PNO, store pair affiliation
                }
            }

            if (madness_process.world->rank() == 0)
                std::cout << "done " << "\n";

            std::string current_label = (pairs.type == MP2_PAIRTYPE) ? "MP2" : "CISPD_Ex" + std::to_string(pairs.cis.number);
            std::vector<std::tuple<double, real_function_3d, std::pair<size_t, size_t>, std::string>> zipped;

            for (auto i = 0; i < all_current_pnos.size(); ++i) {
                zipped.push_back(std::make_tuple(all_current_occ[i], all_current_pnos[i], all_current_ids[i], current_label));
            }

            std::sort(zipped.begin(), zipped.end(),
                      [](const auto& i, const auto& j) { return std::get<0>(i) > std::get<0>(j); });
            if (madness_process.world->rank() == 0)
                std::cout << "sorted " << "\n";

            auto limit = std::min(npno, zipped.size());
            for (auto i = 0; i < limit; ++i) {
                if (pairs.type == MP2_PAIRTYPE) {
                    mp2_pnos.push_back(std::get<1>(zipped[i]));
                    mp2_occ.push_back(std::get<0>(zipped[i]));
                    mp2_ids.push_back(std::get<2>(zipped[i]));
                    mp2_labels.push_back(std::get<3>(zipped[i]));
                } else if (pairs.type == CISPD_PAIRTYPE) {
                    cispd_pnos.push_back(std::get<1>(zipped[i]));
                    cispd_occ.push_back(std::get<0>(zipped[i]));
                    cispd_ids.push_back(std::get<2>(zipped[i]));
                    cispd_labels.push_back(std::get<3>(zipped[i]));
                }
            }
            if (madness_process.world->rank() == 0)
                std::cout << "unzipped " << "\n";
        }

        vecfuncT obs_pnos;
        std::vector<double> occ;
        std::vector<std::pair<size_t, size_t>> pno_ids;
        std::vector<std::string> labels;

        // insert mp2 pnos first, then cis and cispd pnos (if any), then orthogonalize all together
        obs_pnos.insert(obs_pnos.end(), mp2_pnos.begin(), mp2_pnos.end());
        occ.insert(occ.end(), mp2_occ.begin(), mp2_occ.end());
        pno_ids.insert(pno_ids.end(), mp2_ids.begin(), mp2_ids.end());
        labels.insert(labels.end(), mp2_labels.begin(), mp2_labels.end());

        // insert cis x functions
        for (auto& xfct : cis_x_functions) {
            obs_pnos.push_back(xfct);
            occ.push_back(1.0);
            pno_ids.push_back({0, 0});
            labels.push_back("CIS_X");
        }

        // insert cispd pnos after mp2 pnos
        obs_pnos.insert(obs_pnos.end(), cispd_pnos.begin(), cispd_pnos.end());
        occ.insert(occ.end(), cispd_occ.begin(), cispd_occ.end());
        pno_ids.insert(pno_ids.end(), cispd_ids.begin(), cispd_ids.end());
        labels.insert(labels.end(), cispd_labels.begin(), cispd_labels.end());

        if (madness_process.world->rank() == 0)
            std::cout << "collected " << obs_pnos.size() << " pnos" << "\n";
        if (madness_process.world->rank() == 0)
            std::cout << "and " << reference.size() << " reference orbitals" << "\n";

        madness::QProjector<double, 3> Q(*(madness_process.world), reference);
        obs_pnos = Q(obs_pnos);

        if (obs_pnos.size() > 1) {
            if (madness_process.world->rank() == 0)
                std::cout << "Orthonormalizing via Integrals class (mixed method)..." << std::endl;

            Integrals<3> helper(madness_process);
            std::vector<SavedFct<3>> tmp_saved;
            for(auto& f : obs_pnos) tmp_saved.push_back(SavedFct<3>(f));
            nb::ndarray<nb::numpy, double, nb::ndim<1>> occ_ndarray;
            auto ortho_saved = helper.orthonormalize(tmp_saved, "symmetric", 1e-7, occ_ndarray, 1e-6);
            obs_pnos.clear();
            for(auto& s : ortho_saved) obs_pnos.push_back(madness_process.loadfct(s));
        }

        if (occ.size() > obs_pnos.size()) {
            occ.resize(obs_pnos.size());
            pno_ids.resize(obs_pnos.size());
            labels.resize(obs_pnos.size());
             if (madness_process.world->rank() == 0)
                std::cout << "Warning: More occupation numbers than PNOs, resizing to " << obs_pnos.size() << "\n";
        }

        this->labels = std::vector<std::string>(reference.size(), "HF");
        this->labels.insert(this->labels.end(), labels.begin(), labels.end());

        vecfuncT xbasis = reference;


        if (madness_process.world->rank() == 0)
            std::cout << "Forming basis with " << xbasis.size() << " orbitals" << "\n";
        xbasis.insert(xbasis.end(), obs_pnos.begin(), obs_pnos.end());
        if (madness_process.world->rank() == 0)
            std::cout << "filled up to " << xbasis.size() << " orbitals" << "\n";

        this->basis = xbasis;

        // save occ and pno ids
        // fill up with the hf orbitals first
        if (madness_process.world->rank() == 0)
            std::cout << "currently " << occ.size() << " occupation numbers" << "\n";
        if (madness_process.world->rank() == 0)
            std::cout << "currently " << pno_ids.size() << " pno ids" << "\n";

        std::vector<double> tmpx(reference.size(), 2.0);
        tmpx.insert(tmpx.end(), occ.begin(), occ.end());
        occ = tmpx;

        std::vector<std::pair<size_t, size_t>> tmpy;
        for (size_t k = 0; k < reference.size(); k++)
            tmpy.push_back(std::make_pair(k, k));
        tmpy.insert(tmpy.end(), pno_ids.begin(), pno_ids.end());
        pno_ids = tmpy;

        if (madness_process.world->rank() == 0)
            std::cout << "currently " << occ.size() << " occupation numbers" << "\n";
        if (madness_process.world->rank() == 0)
            std::cout << "currently " << pno_ids.size() << " pno ids" << "\n";

        this->occ = occ;
        this->ids = pno_ids;

        nfreeze = pno.param.freeze();
        nemo->get_calc()->reset_aobasis("sto-3g");
        sto3g = nemo->get_calc()->project_ao_basis(*(madness_process.world), nemo->get_calc()->aobasis);
        }

    std::vector<SavedFct<3>> get_pnos_filtered(const std::string& type_filter = "") const {
        std::vector<SavedFct<3>> filtered_pnos;
        for (size_t i = 0; i < basis.size(); ++i) {
            if (labels[i].find(type_filter) != std::string::npos) {
                SavedFct<3> pnorb(basis[i]);

                size_t offset = (labels[i] == "HF" || labels[i] == "CIS_X") ? 0 : this->nfreeze;

                pnorb.info = "type=" + labels[i] + " occ=" + std::to_string(occ[i]) + " ";
                pnorb.info += "pair1=" + std::to_string(ids[i].first + offset) + " ";
                pnorb.info += "pair2=" + std::to_string(ids[i].second + offset) + " ";

                filtered_pnos.push_back(pnorb);
            }
        }
        return filtered_pnos;
    }

    std::vector<SavedFct<3>> get_gs_orbs() const {
        auto hf = get_pnos_filtered("HF");
        auto mp2 = get_pnos_filtered("MP2");

        hf.insert(hf.end(), mp2.begin(), mp2.end());
        return hf;
    }

    std::vector<SavedFct<3>> get_ex_orbs() const {
        auto x_orbs = get_pnos_filtered("CIS_X");
        auto cispd_orbs = get_pnos_filtered("CISPD");

        x_orbs.insert(x_orbs.end(), cispd_orbs.begin(), cispd_orbs.end());
        return x_orbs;
    }

    double get_nuclear_repulsion() const { return nuclear_repulsion; }

    std::vector<SavedFct<3>> get_sto3g() const {
        std::vector<SavedFct<3>> result;
        for (auto x : sto3g)
            result.push_back(SavedFct<3>(x, "atomic"));
        return result;
    }

  private:
    MadnessProcess<3>& madness_process;

    commandlineparser parser;
    vecfuncT basis;
    real_function_3d Vnuc;
    vecfuncT sto3g;
    std::vector<real_function_3d> cis_x_functions;

  protected:
    size_t nfreeze;
    double nuclear_repulsion;
    std::vector<double> occ;
    std::vector<std::pair<size_t, size_t>> ids;
    std::vector<std::string> labels;

  public:
    std::size_t get_frozen_core_dim() const { return this->nfreeze; }
};
