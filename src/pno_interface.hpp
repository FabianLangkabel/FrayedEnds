#pragma once

#include "functionsaver.hpp"
#include <iomanip>
#include <madness/mra/vmra.h>
#include<madness/chem/SCF.h>
#include<madness/chem/nemo.h>
#include<madness/chem/PNO.h>
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
#include "MadnessProcess.hpp"

using namespace madness;
namespace nb = nanobind;

// DEFINE PARAMETER TAGS FOR THE INPUT FILE
const std::string TAG_PNO = "pno";
const std::string TAG_F12 = "f12";
const std::string TAG_PNOInt = "pnoint";
const std::string TAG_CP = "computeprotocol";


template<typename T1, typename T2>
std::ostream& operator << (std::ostream& os, const std::pair<T1,T2>& v){
    os << "(" << v.first << "," << v.second << ")";
    return os;
}

template<typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v){
    os << "[";
    for(auto i=0; i<v.size();++i){
        os << v[i] << " ";
    }
    os << "]";
    return os;
}

// Function for orthogonalization of a basis
std::vector<real_function_3d> orthonormalize_basis(const std::vector<real_function_3d> &basis, const std::string orthogonalization,
		const double thresh, World& world, const bool print_out) {

	// compute overlap, to be passed in orthonormalization routines and potentially printed
	auto S = madness::matrix_inner(world, basis, basis, true);

	auto out_basis = basis;
	if (orthogonalization == "cholesky"){
		out_basis = madness::orthonormalize_cd(basis, S);
	}else if(orthogonalization == "symmetric"){
		out_basis = madness::orthonormalize_symmetric(basis, S);
	}else if(orthogonalization == "canonical") {
		out_basis = madness::orthonormalize_canonical(basis, S, thresh);
	}else if(orthogonalization == "rr_cholesky") {
		if(world.rank()==0) std::cout << std::endl <<  "Be cautious:" << std::endl;
		if(world.rank()==0) std::cout << "Rank reduced cholesky changes ordering in basis via pivoting. Make sure, that this does not interfere with your application (e.g. active-space)." << std::endl;
		out_basis = madness::orthonormalize_rrcd(basis, S, thresh);
	}else{
		MADNESS_EXCEPTION("unknown orthonormalization method", 1);
	}

	if (print_out) {
		const auto new_S = madness::matrix_inner(world, out_basis, out_basis);
		if(world.rank()==0) {
			std::cout << "Overlap Matrix before " << orthogonalization << "\n";
			std::cout << S;
			std::cout << "Overlap Matrix after " << orthogonalization << "\n";
			std::cout << new_S;
		}
	}

	return out_basis;
}
//PNOIntegrals class
class PNOIntParameters: public PNOParameters {
	/*
	 * parameters we need:
	 * == orthogonalization method
	 * == cabs-file (don't pick the one from F12 I guess)
	 * == cherry-pick orbs
	 * == size of desired PNO-basis aka size-OBS
	 * == size of PNO-CABS
	 * == maybe some kind of thresh for cabs? aka cabs_thresh, that is unused anyhow
	 *
	 *
	 *
	 */
	public:
		PNOIntParameters(const PNOParameters& param) : PNOParameters(param){
			initialize_pnoint_parameters();
			set_derived_values(param);
		}
	
		PNOIntParameters(World& world, const commandlineparser& parser, const PNOParameters& param, const std::string& TAG="pnoint") : PNOParameters(param){
			initialize_pnoint_parameters();
			QCCalculationParametersBase::read_input_and_commandline_options(world,parser,TAG);
			set_derived_values(param);
		}
	
	
		void initialize_pnoint_parameters() {
			initialize<std::string>("orthog", "cholesky", "orthogonalization method for PNO basis, also current default for a potential CABS. options: symmetric, canonical, cholesky, rr_cholesky,  ");
			initialize<std::string>("orthog_cabs", "default", "orthogonalization method for CABS basis. options: symmetric, canonical, cholesky, rr_cholesky, uses same as orbitals basis if default.");
			initialize<int>("n_pno", 10, "desired size of PNO-basis (orbital space)");
			initialize<std::string>("cabs_option", "none", "type of complementary auxiliary basis to be used. can be none ~ no CABS, gbs ~ Gaussian basis set as CABS, requires input via auxbas_file, pno ~ use superfluous PNOs as CABS, mixed ~ gbs&pno-option");
			initialize<int>("pno_cabs_size", -1, "size of a potential pno-cabs. defaults to -1 ~ all remaining PNOs, if pno-cabs or mixed-cabs");
			initialize<bool>("only_diag", false, "use only diagonal elements PNOs");
			initialize<std::vector<int> >("cherry_pick", std::vector<int>(), "cherry pick option to choose particular set of PNOs, not just blindly by occupation number. can be useful, if only a restricted number of PNOs can be used, to avoid separating degenerate pairs. can be envoked by passing a vector via [ elem1, elem2, ... ]");
			initialize<std::vector<int> >("ignore_pair", std::vector<int>(), "ignore certain pairs in the selection of pnos numbering is 00-0, 01-1, 02-2 ... ");
			initialize<double>("gamma",1.4, "The f12 length scale, here to be used to compute f12-integrals");
			initialize<double>("cabs_thresh",1.e-4, " thresh for cabs part ");
			initialize<std::string>("auxbas", "none", "atom centered partial wave guess of format like 'h-2s1p-o-3s2p1d' ");
			initialize<std::string>("auxbas_file", "none", "Use external comp. aux. basis in addition to the pnos as auxbasis. Give the filename as parameter. Give the auxbas in turbomole format. Don't use contractions. If a file is specified the auxbas parameter has no effect");
			initialize<bool>("print_pno_overlap", true, "Print overlap matrix at certain steps in computation, for debugging purposes.");
			initialize<bool>("hardcore_boson", false, "compute only integrals for hardcore boson hamiltonian (restricted to double occupations)");
		}
	
		std::string orthogonalization()const { return get<std::string>("orthog");}
		std::string cabs_orthogonalization()const { 
			if (get<std::string>("orthog_cabs") == "default")
				return orthogonalization();
			else
				return get<std::string>("orthog_cabs");
		}
	
		bool hardcore_boson()const{return get<bool>("hardcore_boson");}
		std::string cabs_option()const { return get<std::string>("cabs_option");}
		int n_pno()const { return get<int >("n_pno");}
		int pno_cabs_size()const { return get<int >("pno_cabs_size");}
		bool only_diag()const { return get<bool>("only_diag"); }
		std::vector<int> cherry_pick()const { 
			return get<std::vector<int> >("cherry_pick");
		}
		std::vector<int> ignore_pair()const { 
			return get<std::vector<int> >("ignore_pair");
		}
		double cabs_thresh()const { return get<double >("cabs_thresh");}
		std::string auxbas_file()const {
			return get<std::string >("auxbas_file");
		}
		double gamma()const { return get<double >("gamma");}
	
		// TODO guess we don't need that here...
		std::map<std::string,std::vector<int> > auxbas()const {
			return partial_wave("auxbas");
		}
		bool print_pno_overlap()const { return get<bool>("print_pno_overlap"); }
	
		void set_derived_values(const PNOParameters& param) {
			set_derived_value("cabs_thresh", thresh());
			set_derived_value("n_pno", std::min(10, param.maxrank()));
					set_derived_value("only_diag", diagonal());
		}
	
};
	
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


class PNOInterface: public MadnessProcess{
	public:
		PNOInterface(std::string argv, double L, long k, double thresh, int initial_level, int truncate_mode, bool refine, int n_threads) : MadnessProcess(L, k, thresh, initial_level, truncate_mode, refine, n_threads)
		{
			
			auto [argc, charArray] = stringToCharPointerArray(argv);
			parser = commandlineparser(argc,charArray);
			freeCharPointerArray(charArray,argc);
		}
		~PNOInterface()
		{	
			basis.clear();
			Vnuc.clear();
		}

		SavedFct get_nuclear_potential(){
			return SavedFct(Vnuc);
		}

		void DeterminePNOsAndIntegrals() 
		{	
			std::cout.precision(6);
			if(world->rank()==0){
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
					Nemo nemo(*world,parser);
					nemo.get_calc()->param.print();
			const double scf_energy = nemo.value();
			if (world->rank() == 0) print("nemo energy: ", scf_energy);
			if (world->rank() == 0) printf(" at time %.1f\n", wall_time());
			const double time_scf_end = wall_time();
			// assert that no nemo corrfactor is actually used (not yet supported in PNO-MP2)
			if(nemo.ncf->type() != madness::NuclearCorrelationFactor::None){
				MADNESS_EXCEPTION("Nuclear Correlation Factors not yet supported in MRA-PNOs. Add ncf (none,1.0) to your dft input",1);
			}
			{
			Vnuc = nemo.ncf -> U2();
			nuclear_repulsion = nemo.get_calc()->molecule.nuclear_repulsion_energy();
			}
	
			// Compute MRA-PNO-MP2-F12
			const double time_pno_start = wall_time();
			PNOParameters parameters(*world,parser,nemo.get_calc()->molecule,TAG_PNO);
			F12Parameters paramf12(*world, parser, parameters, TAG_F12);
			PNOIntParameters paramsint(*world, parser, parameters, TAG_PNOInt); //should be removed eventually
			paramsint.print("PNO Integrals evaluated as:\npnoint","end"); //ditto
			PNO pno(*world, nemo, parameters, paramf12);
			pno.solve();
			const double time_pno_end = wall_time();
	
	
			if(world->rank()==0){
				std::cout << std::setfill(' ');
				std::cout << "\n\n\n";
				std::cout << "--------------------------------------------------\n";
				std::cout << "MRA-PNO-MP2-F12 ended \n";
				std::cout << "--------------------------------------------------\n";
				std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
				std::cout << std::setw(25) << "energy scf" << " = " << scf_energy << "\n";
				std::cout << "--------------------------------------------------\n";
			}
	
			if(world->rank()==0){
				std::cout << "restarting PNO to reload pairs that converged before and were frozen\n";
			}
			pno.param.set_user_defined_value<std::string>("restart", "all");
			pno.param.set_user_defined_value<std::string>("no_opt", "all");
			pno.param.set_user_defined_value<std::string>("no_guess", "all");
			pno.param.set_user_defined_value<std::string>("adaptive_solver", "none");
			std::vector<PNOPairs> all_pairs;
			pno.solve(all_pairs);
	
			double mp2_energy = 0.0;
			if(world->rank()==0) std::cout<< std::setw(25) << "time pno" << " = " << time_pno_end - time_pno_start << "\n";
			for(const auto& pairs: all_pairs){
				if(pairs.type == MP2_PAIRTYPE){
					mp2_energy = pairs.energies.total_energy();
				}
				std::pair<size_t, size_t> ranks= pno.get_average_rank(pairs.pno_ij);
				if(world->rank()==0){
					std::string name;
					std::stringstream ss;
					ss << pairs.type;
					ss >> name;
					std::cout<< std::setw(25) << "energy "+name << " = " << pairs.energies.total_energy() << "\n";
					std::cout<< std::setw(25) << "average pno rank " + name << " = " << ranks.first << "\n";
					std::cout<< std::setw(25) << "max pno rank " + name << " = " << ranks.second << "\n";
				}
			}
			if(world->rank()==0 and mp2_energy != 0.0){
				std::cout << "--------------------------------------------------\n";
				std::cout<< std::setw(25) << "energy(total)" << " = " << scf_energy + mp2_energy << "\n";
				std::cout << "--------------------------------------------------\n";
				std::cout << "\n\n\n";
			}
	
			// compute orthogonalized mp2 basis and print out hamiltonian tensors
			std::cout << std::setprecision(8);
			std::cout << std::fixed;
			std::cout << std::showpos;
	
			const std::string orthogonalization = paramsint.orthogonalization(); // need to get ortho method elsewhere
			if (world->rank()==0) std::cout << "Orthonormalization technique used: " << orthogonalization << std::endl;
			const bool orthogonalize = orthogonalization != "none";
			const double h_thresh = 1.e-7; // neglect integrals
			double thresh = parameters.thresh();
	
			const std::string cabs_option = paramsint.cabs_option();
			bool cabs_switch = false;
			if (cabs_option=="pno" || cabs_option=="gbs" || cabs_option=="mixed")
				cabs_switch = true;
			if (world->rank()==0) {
				if (cabs_switch) std::cout << "CABS option " << cabs_option << std::endl;
				else if (!cabs_switch) std::cout << "No CABS used." << std::endl;
			}
			const bool only_diag = paramsint.only_diag();
			const int basis_size = paramsint.n_pno();
			int pno_cabs_size = paramsint.pno_cabs_size();
	
			thresh = std::min(thresh, 1.e-4);
			if(world->rank()==0) std::cout << "Tightening thresholds to " << thresh << " for post-processing\n";
			FunctionDefaults<3>::set_thresh(thresh);
	
			vecfuncT reference = nemo.get_calc()->amo;
			vecfuncT obs_pnos;
			std::vector<real_function_3d> rest_pnos;
			std::vector<double> occ;
			std::vector<double> rest_occ;
			std::vector<std::pair<size_t,size_t>> pno_ids;
			std::vector<std::pair<size_t,size_t>> rest_ids;
			std::string name;

			for(auto& pairs: all_pairs){
							// list of pairs to ignore:
							std::vector<bool> ignore_pair(pairs.npairs, false);
							for(auto x : paramsint.ignore_pair()){
								ignore_pair[x] = true;
								if(world->rank()==0) std::cout << "will ignore pair " << x << "\n";
							} 
	
				const auto& pno_ij = pairs.pno_ij;
				const auto& rdm_evals = pairs.rdm_evals_ij;
				const bool is_gs = pairs.type == MP2_PAIRTYPE;
	
				if (not is_gs){
					const auto& x = pairs.cis.x;
					reference.insert(reference.end(), x.begin(), x.end());
					name = "molecule_ex_" + std::to_string(pairs.cis.number);
				}else{
					name = "molecule";
				}
	
				std::vector<real_function_3d> all_current_pnos;
				// collect PNOs from all pairs and sort by occupation number, keeping pair information via name
				for(ElectronPairIterator it=pno.pit();it;++it){
					if (only_diag and not it.diagonal()){
						if(world->rank()==0) std::cout << "skipping pair (not diagonal) " << it.name() << "\n";
						continue;
									}else if(ignore_pair[it.ij()]){
						if(world->rank()==0) std::cout << "skipping pair (demanded by input) " << it.name() << "\n";
						continue;
					}else{
						if(world->rank()==0) std::cout << "adding " << it.name() << "\n";
						const auto& pair = pno_ij[it.ij()];
						all_current_pnos.insert(all_current_pnos.end(), pair.begin(), pair.end());
						for (auto ii=0; ii<rdm_evals[it.ij()].size();++ii){
							occ.push_back(rdm_evals[it.ij()][ii]);
							pno_ids.push_back(std::make_pair(it.i(),it.j()));  // for each eigenvalue ~ PNO, store pair affiliation
						}
					}
				}
	            if(world->rank()==0) std::cout << "done " << "\n";
				std::vector<std::tuple<double, real_function_3d, std::pair<size_t,size_t> > > zipped;
				for (auto i=0; i< all_current_pnos.size(); ++i){
					zipped.push_back(std::make_tuple(occ[i], all_current_pnos[i], pno_ids[i]));
				}
	
				std::sort(zipped.begin(), zipped.end(), [](const auto& i, const auto& j) { return std::get<0>(i) > std::get<0>(j); });
	            if(world->rank()==0) std::cout << "sorted " << "\n";

				std::vector<double> unzipped_first;
				std::vector<real_function_3d> unzipped_second;
				std::vector<std::pair<size_t,size_t> > unzipped_third;
				for (auto i=0; i<basis_size;++i){
					unzipped_first.push_back(std::get<0>(zipped[i]));
					unzipped_second.push_back(std::get<1>(zipped[i]));
					unzipped_third.push_back(std::get<2>(zipped[i]));
				}
				if(world->rank()==0) std::cout << "unzipped " << "\n";

				occ = unzipped_first;
				all_current_pnos = unzipped_second;
				pno_ids = unzipped_third;

				obs_pnos.insert(obs_pnos.end(), all_current_pnos.begin(), all_current_pnos.end());
			}

	        if(world->rank()==0) std::cout << "collected " << obs_pnos.size() << " pnos" << "\n";
            if(world->rank()==0) std::cout << "and " << reference.size() << " reference orbitals" << "\n";

			// reference projector (not automatically fullfilled for CIS)
			// projection is to keep the reference and CIS orbitals untouched in the orthogonalization
			madness::QProjector<double, 3> Q(*world, reference);
			obs_pnos = Q(obs_pnos);
			vecfuncT xbasis = reference;
			if(world->rank()==0) std::cout << "Forming basis with " << xbasis.size() << " orbitals" << "\n";
			xbasis.insert(xbasis.end(), obs_pnos.begin(), obs_pnos.end());
			if(world->rank()==0) std::cout << "filled up to " << xbasis.size() << " orbitals" << "\n";

			this->basis = xbasis;

			// save occ and pno ids
			// fill up with the hf orbitals first
			if(world->rank()==0) std::cout << "currently " << occ.size() << " occupation numbers" << "\n";
			if(world->rank()==0) std::cout << "currently " << pno_ids.size() << " pno ids" << "\n";

			std::vector<double> tmpx(reference.size(),2.0);
			tmpx.insert(tmpx.end(), occ.begin(), occ.end());
			occ = tmpx;

			std::vector<std::pair<size_t,size_t>> tmpy(reference.size(), std::make_pair(9999,9999));
			tmpy.insert(tmpy.end(), pno_ids.begin(), pno_ids.end());
			pno_ids = tmpy;

			if(world->rank()==0) std::cout << "currently " << occ.size() << " occupation numbers" << "\n";
			if(world->rank()==0) std::cout << "currently " << pno_ids.size() << " pno ids" << "\n";


			this->occ = occ;
			this->ids = pno_ids;
		}

		std::vector<SavedFct> GetPNOs(int core_dim, int as_dim, int froz_virt_dim) const
		{	
			if (core_dim+as_dim+froz_virt_dim != basis.size()){
				std::cerr << "PNOInterface::GetPNOs: core_dim + as_dim + froz_virt_dim != basis.size() " << core_dim << " "<< as_dim << " " << froz_virt_dim << " " << basis.size() << std::endl;
			}
			std::vector<SavedFct> pnos;
			for (auto i=0; i<basis.size(); ++i){
				SavedFct pnorb(basis[i]);
				if (i<core_dim){
					pnorb.type="frozen_occ";
					pnorb.info="occ="+std::to_string(occ[i])+" ";
				}else if (i<core_dim+as_dim){
					pnorb.type="active";
					pnorb.info="occ="+std::to_string(occ[i])+" ";
					pnorb.info+="idx="+std::to_string(ids[i].first)+" ";
					pnorb.info+="idy="+std::to_string(ids[i].second)+" ";
				}else if (i<core_dim+as_dim+froz_virt_dim){
					pnorb.type="frozen_virt";
					pnorb.info="occ="+std::to_string(occ[i])+" ";
					pnorb.info+="idx="+std::to_string(ids[i].first)+" ";
					pnorb.info+="idy="+std::to_string(ids[i].second)+" ";
				}
				pnos.push_back(pnorb);
			}
			return pnos;
		}
		nb::ndarray<nb::numpy, double, nb::ndim<2> > GetHTensor()
		{
			nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(h.ptr(), {basis.size(), basis.size()});
			return numpy_array;
		}
		nb::ndarray<nb::numpy, double, nb::ndim<4> > GetGTensor()
		{
			nb::ndarray<nb::numpy, double, nb::ndim<4> > numpy_array(g.ptr(), {basis.size(), basis.size(), basis.size(), basis.size()});
			return numpy_array;
		}
		nb::ndarray<nb::numpy, double, nb::ndim<4> > GetF12Tensor() //what is the f tensor?
		{
			nb::ndarray<nb::numpy, double, nb::ndim<4> > numpy_array(f.ptr(), {basis.size(), basis.size(), basis.size(), basis.size()});
			return numpy_array;
		}
		double GetNuclearRepulsion() const
		{
			return nuclear_repulsion;
		}

	private:
		commandlineparser parser;
		vecfuncT basis;
		std::vector<double> occ;
		std::vector<std::pair<size_t,size_t>> ids;
		madness::Tensor<double> g;
		madness::Tensor<double> f;
		madness::Tensor<double> h;
		double nuclear_repulsion;
		real_function_3d Vnuc;
};




