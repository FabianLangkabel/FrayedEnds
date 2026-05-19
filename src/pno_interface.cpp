#include "pno_interface.hpp"
#include <iomanip>
#include <sstream>
#include <cstring>
#include <fstream>
#include <regex>
#include <algorithm>
#include <madness/chem/SCF.h>
#include <madness/chem/TDHF.h>
#include <madness/chem/PNO.h>

using namespace madness;

// DEFINE PARAMETER TAGS FOR THE INPUT FILE
const std::string TAG_PNO = "pno";
const std::string TAG_F12 = "f12";
const std::string TAG_CP = "computeprotocol";

PNOInterface::PNOInterface(MadnessProcess<3>& mp, std::string argv) : madness_process(mp) {
    auto [argc, charArray] = stringToCharPointerArray(argv);
    parser = commandlineparser(argc, charArray);
    freeCharPointerArray(charArray, argc);
}

PNOInterface::~PNOInterface() {
    basis.clear();
    Vnuc.clear();
}

SavedFct<3> PNOInterface::get_nuclear_potential() { 
    return SavedFct<3>(Vnuc); 
}

void PNOInterface::compute_scf() {
    if (madness_process.world->rank() == 0) {
        std::cout << "\n\n";
        std::cout << "-------------------------------------------------------------------------------------\n";
        std::cout << "--- Starting SCF Reference Calculation ---\n";
        std::cout << "-------------------------------------------------------------------------------------\n";
        std::cout << "\n\n";
    }

    const double time_scf_start = wall_time();
    auto nemo = std::make_shared<Nemo>(*(madness_process.world), parser);
    nemo->get_calc()->param.print();
    this->scf_energy = nemo->value();

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

    this->hf_orbitals = nemo->get_calc()->amo; // store HF orbitals for later use
    this->Vnuc = nemo->ncf->U2();
    this->nuclear_repulsion = nemo->get_calc()->molecule.nuclear_repulsion_energy();
    this->nemo = nemo; // store nemo for later use

    if (madness_process.world->rank() == 0) {
        std::cout << "\n\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "--- SCF Calculation ended --- \n";
        std::cout << "--------------------------------------------------\n";
        std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
        std::cout << std::setw(25) << "energy scf" << " = " << scf_energy << "\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "\n\n";
    }
}

void PNOInterface::run(const size_t basis_size) {
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

    // Compute SCF reference
    if (!nemo) compute_scf();

    // Compute MRA-PNO-MP2-F12
    const double time_pno_start = wall_time();
    PNOParameters parameters(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
    F12Parameters paramf12(*(madness_process.world), parser, parameters, TAG_F12);
    PNO pno(*(madness_process.world), *nemo, parameters, paramf12);
    pno.solve();
    const double time_pno_end = wall_time();

    if (madness_process.world->rank() == 0) {
        std::cout << std::setfill(' ');
        std::cout << "\n\n\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "MRA-PNO-MP2 ended \n";
        std::cout << "--------------------------------------------------\n";
        std::cout << std::setw(25) << "time pno" << " = " << time_pno_end - time_pno_start << "\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "restarting PNO to reload all pairs that converged before and were frozen\n";
    }

    pno.param.set_user_defined_value<std::string>("restart", "all");
    pno.param.set_user_defined_value<std::string>("no_opt", "all");
    pno.param.set_user_defined_value<std::string>("no_guess", "all");
    pno.param.set_user_defined_value<std::string>("adaptive_solver", "none");

    std::vector<PNOPairs> mp2_pairs;
    pno.solve(mp2_pairs); // solve MP2 again to reload all pairs that converged before and were frozen

    double mp2_energy = 0.0;

    for (const auto& pairs : mp2_pairs) {
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

    const size_t npno = basis_size - hf_orbitals.size(); 
    std::cout << "basis size requested: " << basis_size << "\n";
    std::cout << "reference size: " << hf_orbitals.size() << "\n";
    std::cout << "number of PNOs requested: " << npno << "\n";

    std::vector<real_function_3d> mp2_pnos;
    std::vector<double> mp2_occ;
    std::vector<std::pair<size_t, size_t>> mp2_ids;
    std::vector<std::string> mp2_labels;

    for (auto& pairs : mp2_pairs) {
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
                std::cout << "adding " << it.name() << "\n";

            const auto& pair = pno_ij[it.ij()];
            all_current_pnos.insert(all_current_pnos.end(), pair.begin(), pair.end());
            for (auto ii = 0; ii < rdm_evals[it.ij()].size(); ++ii) {
                all_current_occ.push_back(rdm_evals[it.ij()][ii]);
                all_current_ids.push_back(std::make_pair(it.i(), it.j())); // for each eigenvalue ~ PNO, store pair affiliation
            }
        }

        if (madness_process.world->rank() == 0)
            std::cout << "done " << "\n";

        const std::string current_label = "MP2";
        std::vector<std::tuple<double, real_function_3d, std::pair<size_t, size_t>, std::string>> zipped;

        for (auto i = 0; i < all_current_pnos.size(); ++i) {
            zipped.push_back(std::make_tuple(all_current_occ[i], all_current_pnos[i], all_current_ids[i], current_label));
        }

        std::sort(zipped.begin(), zipped.end(), [](const auto& i, const auto& j) { return std::get<0>(i) > std::get<0>(j); });
        if (madness_process.world->rank() == 0)
            std::cout << "sorted " << "\n";

        const size_t n_take = std::min(npno, zipped.size()); // take only as many PNOs as requested by npno, but not more than available

        for (auto i = 0; i < n_take; ++i) {
            mp2_occ.push_back(std::get<0>(zipped[i]));
            mp2_pnos.push_back(std::get<1>(zipped[i]));
            mp2_ids.push_back(std::get<2>(zipped[i]));
            mp2_labels.push_back(std::get<3>(zipped[i]));
        }
        if (madness_process.world->rank() == 0)
            std::cout << "unzipped " << "\n";
    }

    if (madness_process.world->rank() == 0) 
        std::cout << "collected " << mp2_pnos.size() << " mp2 pnos" << "\n";
    if (madness_process.world->rank() == 0)
        std::cout << "and " << hf_orbitals.size() << " reference orbitals" << "\n";

    madness::QProjector<double, 3> Q(*(madness_process.world), hf_orbitals); // projector to ensure orthogonality to reference (HF orbitals)
    mp2_pnos = Q(mp2_pnos); //project out reference (HF) from pnos to ensure orthogonality

    this->basis = hf_orbitals; // start with reference (HF) orbitals as basis
    this->basis.insert(this->basis.end(), mp2_pnos.begin(), mp2_pnos.end()); // add pnos to basis after reference (HF) orbitals

    if (madness_process.world->rank() == 0) {
        std::cout << "Forming basis with " << this->basis.size() << " orbitals" << "\n";
        std::cout << "filled up to " << this->basis.size() << " orbitals" << "\n";
    }

    this->labels = std::vector<std::string>(hf_orbitals.size(), "HF");
    this->labels.insert(this->labels.end(), mp2_labels.begin(), mp2_labels.end());

    // save occ and pno ids
    // fill up with the hf orbitals first
    if (madness_process.world->rank() == 0) {
        std::cout << "currently " << mp2_occ.size() << " occupation numbers" << "\n";
        std::cout << "currently " << mp2_ids.size() << " mp2 pno ids" << "\n";
    }

    std::vector<double> tmpx(hf_orbitals.size(), 2.0); // assign occupation number 2.0 to HF orbitals
    tmpx.insert(tmpx.end(), mp2_occ.begin(), mp2_occ.end());
    this->occ = tmpx; // insert occ of HF orbitals at the beginning

    std::vector<std::pair<size_t, size_t>> tmpy;
    for (size_t k = 0; k < hf_orbitals.size(); k++)
        tmpy.push_back(std::make_pair(k, k));
    tmpy.insert(tmpy.end(), mp2_ids.begin(), mp2_ids.end());
    this->ids = tmpy; // insert pair ids for HF orbitals at the beginning

    if (madness_process.world->rank() == 0) {
        std::cout << "currently " << this->occ.size() << " occupation numbers" << "\n";
        std::cout << "currently " << this->ids.size() << " pno ids" << "\n";
    }

    nfreeze = pno.param.freeze();
    nemo->get_calc()->reset_aobasis("sto-3g");
    sto3g = nemo->get_calc()->project_ao_basis(*(madness_process.world), nemo->get_calc()->aobasis);

    this->mp2_pairs = std::move(mp2_pairs); // store all pairs 
    this->mp2_pnos = std::move(mp2_pnos); // store mp2 pnos 
    this->mp2_computed = true; // flag if mp2 pnos were computed and stored successfully
}

void PNOInterface::compute_cis(const size_t n_excitations) {
    if (!nemo) {
        MADNESS_EXCEPTION("Error: compute_cis() called before run(). Nemo not initialized. Run compute_scf() or run() first.", 1);
    }

    cis_computed = false; // reset flag at the beginning of compute_cis

    if (n_excitations == 0) {
        if (madness_process.world->rank() == 0)
            std::cout << "No excitations requested for CIS. Skipping compute_cis.\n";
        return;
    }

    // put cis computation into parser so TDHF can read it 
    parser.set_keyval("tdhf", "nexcitations " + std::to_string(n_excitations) + "; thresh " + std::to_string(1.0e-5) + "; restart no_restart");

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
    cis_x_per_root.resize(cis_roots.size()); 
    const double time_cis_end = wall_time();

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
            cis_x_per_root[ex].push_back(cc_func.function); // store x vector functions for each root
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

    cis_computed = true; 
}

void PNOInterface::compute_cispd(const size_t basis_size) {
    if (!nemo) {
        MADNESS_EXCEPTION("Error: compute_cipsd() called before run(). Nemo not initialized. Run run() first.", 1);
    }
    if (!cis_computed || cis_roots.empty()) {
        MADNESS_EXCEPTION("Error: No CIS roots available. Run compute_cis() first", 1);
    }

    const std::string pno_base = parser.value("pno"); // save original pno key to modify it for cispd and then reset it later for the final pno run
    const size_t n_ex = cis_roots.size();

    std::vector<vecfuncT>                              cispd_pnos_per_ex(cis_roots.size());
    std::vector<std::vector<double>>                   cispd_occ_per_ex(cis_roots.size());
    std::vector<std::vector<std::pair<size_t,size_t>>> cispd_ids_per_ex(cis_roots.size());
    std::vector<std::vector<std::string>>              cispd_labels_per_ex(cis_roots.size());

    std::cout << std::setprecision(8);
    std::cout << std::fixed;
    std::cout << std::showpos;

    const double time_cispd_start = wall_time();

    for (size_t ex = 0; ex < cis_roots.size(); ++ex) {
        // add cispd-specific parameters to parser for excitation calculation
        parser.set_keyval("pno", pno_base + "; cispd_number " + std::to_string(ex) + "; cispd_energy " + std::to_string(cis_roots[ex].omega));

        PNOParameters cispd_parameters(*(madness_process.world), parser, nemo->get_calc()->molecule, TAG_PNO);
        F12Parameters cispd_paramf12(*(madness_process.world), parser, cispd_parameters, TAG_F12);
        PNO pno_cispd(*(madness_process.world), *nemo, cispd_parameters, cispd_paramf12);

        std::vector<PNOPairs> cispd_pairs;

        if (mp2_computed && !mp2_pairs.empty()) { // mp2_pairs should contain at least the mp2 pairs, but check to be sure
            cispd_pairs.push_back(mp2_pairs[0]); // add mp2 as starting point for cispd
        }

        pno_cispd.solve(cispd_pairs);

        std::vector<real_function_3d> all_current_pnos;
        std::vector<double> all_current_occ;
        std::vector<std::pair<size_t, size_t>> all_current_ids;

        for (auto& pairs : cispd_pairs) {
            if (pairs.type != CISPD_PAIRTYPE) {
                continue;
            }
            const auto& pno_ij = pairs.pno_ij;
            const auto& rdm_evals = pairs.rdm_evals_ij;
            const bool only_diag = true;

            for (ElectronPairIterator it = pno_cispd.pit(); it; ++it) {
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
                    all_current_ids.push_back(std::make_pair(it.i(), it.j())); // for each eigenvalue ~ PNO, store pair affiliation
                }
            }
        }

        if (madness_process.world->rank() == 0)
            std::cout << "done " << "\n";

        std::vector<std::tuple<double, real_function_3d, std::pair<size_t, size_t>, std::string>> zipped;

        const std::string current_label = "CISPD_EX" + std::to_string(ex);
        for (auto i = 0; i < all_current_pnos.size(); ++i) {
            zipped.push_back(std::make_tuple(all_current_occ[i], all_current_pnos[i], all_current_ids[i], current_label));
        }

        std::sort(zipped.begin(), zipped.end(), [](const auto& i, const auto& j) { return std::get<0>(i) > std::get<0>(j); });
        if (madness_process.world->rank() == 0)
            std::cout << "sorted " << "\n";

        size_t n_cis_x = (ex < cis_x_per_root.size()) ? cis_x_per_root[ex].size() : 0;
        size_t requested = (basis_size > n_cis_x) ? basis_size - n_cis_x : 0;
        size_t n_take = std::min(requested, zipped.size());

        if (madness_process.world->rank() == 0 && n_take < requested) {
            std::cout << "Warning: requested " << requested
                    << " CISPD PNOs for excitation " << ex
                    << " but only " << zipped.size() << " available.\n";
        }

        for (size_t i = 0; i < n_take; ++i) {
            cispd_pnos_per_ex[ex].push_back(std::get<1>(zipped[i]));
            cispd_occ_per_ex[ex].push_back(std::get<0>(zipped[i]));
            cispd_ids_per_ex[ex].push_back(std::get<2>(zipped[i]));
            cispd_labels_per_ex[ex].push_back("CISPD_EX" + std::to_string(ex));
        }

        this->cispd_pnos.insert(cispd_pnos.end(), cispd_pnos_per_ex[ex].begin(), cispd_pnos_per_ex[ex].end());
        cispd_occ.insert(cispd_occ.end(), cispd_occ_per_ex[ex].begin(), cispd_occ_per_ex[ex].end());
        cispd_ids.insert(cispd_ids.end(), cispd_ids_per_ex[ex].begin(), cispd_ids_per_ex[ex].end());
        cispd_labels.insert(cispd_labels.end(), cispd_labels_per_ex[ex].begin(), cispd_labels_per_ex[ex].end());
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
    parser.set_keyval("pno", pno_base); // reset parser
}

std::vector<SavedFct<3>> PNOInterface::get_hf_orbitals() const {
    std::vector<SavedFct<3>> hf_orbs;
    for (size_t i = 0; i < hf_orbitals.size(); ++i) {
        SavedFct<3> pnorb(hf_orbitals[i]);

        pnorb.info = "type=" + labels[i] + " occ=2.0 pair1=" + std::to_string(i) + 
                  " pair2=" + std::to_string(i);
        hf_orbs.push_back(pnorb);
    }
    return hf_orbs;
}

std::vector<SavedFct<3>> PNOInterface::get_mp2_pnos() const {
    std::vector<SavedFct<3>> mp2_orbs;
    size_t offset = hf_orbitals.size();
    for (size_t i = 0; i < mp2_pnos.size(); ++i) {
        SavedFct<3> pnorb(mp2_pnos[i]);

        pnorb.info = "type=" + labels[i + offset] + " occ=" + std::to_string(occ[offset + i]) + " ";
        pnorb.info += "pair1=" + std::to_string(ids[offset + i].first + nfreeze) + " ";
        pnorb.info += "pair2=" + std::to_string(ids[offset + i].second + nfreeze);

        mp2_orbs.push_back(pnorb);
    }
    return mp2_orbs;
}

// get all ground state orbitals (HF and MP2 PNOs)
std::vector<SavedFct<3>> PNOInterface::get_orbitals() const {
    auto hf = get_hf_orbitals();
    auto mp2 = get_mp2_pnos();

    hf.insert(hf.end(), mp2.begin(), mp2.end());

    if (madness_process.world->rank() == 0) {
        for (const auto& orb : hf) {
            std::cout << "GS Orb info: " << orb.info << "\n";
        }
    }
    return hf;
}

std::vector<std::vector<SavedFct<3>>> PNOInterface::get_cis_x_per_root() const {
    std::vector<std::vector<SavedFct<3>>> result;
    for (size_t ex = 0; ex < cis_x_per_root.size(); ++ex) {
        std::vector<SavedFct<3>> roots;
        for (const auto& func : cis_x_per_root[ex]) {
            SavedFct<3> pnorb(func);
            pnorb.info = "type=CIS_X_EX" + std::to_string(ex) + " occ=1.0 pair1=0 pair2=0";
            roots.push_back(pnorb);
        }

        if (madness_process.world->rank() == 0) {
            for (const auto& orb : roots) {
                std::cout << "CIS X for exitation " << ex <<" info: " << orb.info << "\n";
            }
        }
        
        result.push_back(roots);
    }

    
    return result;
}

std::vector<SavedFct<3>> PNOInterface::get_cispd_orbitals() const {
    std::vector<SavedFct<3>> cispd_orbs;
    for (size_t i = 0; i < cispd_pnos.size(); ++i) {
        SavedFct<3> pnorb(cispd_pnos[i]);
        
        pnorb.info = "type=" + cispd_labels[i] + " occ=" + std::to_string(cispd_occ[i]) + " ";
        pnorb.info += "pair1=" + std::to_string(cispd_ids[i].first + nfreeze) + " ";
        pnorb.info += "pair2=" + std::to_string(cispd_ids[i].second + nfreeze);
        cispd_orbs.push_back(pnorb);
    }

    if (madness_process.world->rank() == 0) {
        for (const auto& orb : cispd_orbs) {
            std::cout << "CISPD info: " << orb.info << "\n";
        }
    }
    return cispd_orbs;
}

std::vector<SavedFct<3>> PNOInterface::get_sto3g() const {
    std::vector<SavedFct<3>> result;
    for (auto x : sto3g)
        result.push_back(SavedFct<3>(x, "atomic"));
    return result;
}
