#include "integrals_open_shell.hpp"

using namespace madness;

template <std::size_t NDIM>
class open_shell_integral_storage {
  public:
    open_shell_integral_storage() {}
    ~open_shell_integral_storage() {}

    void update_integrals(
      Integrals_open_shell<NDIM> &Integrator,
      std::array<std::vector<Function<double, NDIM>>, 2> &core_orbitals, 
      std::array<std::vector<Function<double, NDIM>>, 2> &active_orbitals, 
      std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, 
      std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn, 
      std::array<std::vector<Function<double, NDIM>>, 2> &orbs_aa,
      Function<double, NDIM> &V,
      bool caclulate_core_as_for_eff_ham,
      bool caclulate_core_as_for_as_refinement,
      bool caclulate_core_as_for_core_refinement
    )
    {
        // Calculate one electron Integrals
        as_integrals_one_body = Integrator.compute_potential_integrals(active_orbitals, V);
        {
            std::array<madness::Tensor<double>, 2> kin_Integrals = Integrator.compute_kinetic_integrals(active_orbitals);
            as_integrals_one_body[0] += kin_Integrals[0];
            as_integrals_one_body[1] += kin_Integrals[1];
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        // Calculate two electron Integrals
        as_integrals_two_body = Integrator.compute_two_body_integrals(active_orbitals, orbs_kl, coul_orbs_mn);
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // Calculate Core-AS interaction integrals
        if((core_orbitals[0].size() + core_orbitals[1].size()) > 0)
        {
            core_as_integrals_one_body_ak = Integrator.compute_core_as_integrals_one_body(core_orbitals, active_orbitals, V);
            std::vector<std::vector<madness::Tensor<double>>> core_as_integrals_two_body = Integrator.compute_core_as_integrals_two_body(core_orbitals, active_orbitals, orbs_kl, coul_orbs_mn, orbs_aa, true, true, true, true, true);
            core_as_integrals_two_body_akal = core_as_integrals_two_body[0];
            core_as_integrals_two_body_akla = core_as_integrals_two_body[1];
            core_as_integrals_two_body_akln = core_as_integrals_two_body[2];
            core_as_integrals_two_body_abak = core_as_integrals_two_body[3];
            core_as_integrals_two_body_baak = core_as_integrals_two_body[4];
        }
    }

  inline double kl(int k, int l, int spin_combination) const noexcept {
    return as_integrals_one_body[spin_combination](k, l);
  }

  inline double phys_klmn(int k, int l, int m, int n, int spin_combination) const noexcept {
    return as_integrals_two_body[spin_combination](k, l, m, n);
  }

  inline double ak(int a, int k, int spin_combination) const noexcept {
    return core_as_integrals_one_body_ak[spin_combination](a, k);
  }

  inline double phys_akln(int a, int k, int l, int n, int spin_combination) const noexcept {
    return core_as_integrals_two_body_akln[spin_combination](a, k, l, n);
  }

  inline double phys_akal(int a, int k, int l, int spin_combination) const noexcept {
    return core_as_integrals_two_body_akal[spin_combination](a, k, l);
  }

  inline double phys_akla(int a, int k, int l, int spin_combination) const noexcept {
    return core_as_integrals_two_body_akla[spin_combination](a, k, l);
  }

  inline double phys_abak(int a, int b, int k, int spin_combination) const noexcept {
    return core_as_integrals_two_body_abak[spin_combination](a, b, k);
  }

  inline double phys_baak(int a, int b, int k, int spin_combination) const noexcept {
    return core_as_integrals_two_body_baak[spin_combination](a, b, k);
  }

  public:
    //Active space integrals
    std::array<madness::Tensor<double>, 2> as_integrals_one_body; // (k,l)
    std::array<madness::Tensor<double>, 3> as_integrals_two_body; // (k,l,m,n)

    //Core - AS integrals for AS orbital refinement
    std::array<madness::Tensor<double>, 2> core_as_integrals_one_body_ak;   // (a,k)

    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akln; // (a,k,l,n)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akal; // (a,k,l)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akla; // (a,k,l)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_abak; // (a,b,k)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baak; // (a,b,k)

    //Core - AS integrals for core refinement
};