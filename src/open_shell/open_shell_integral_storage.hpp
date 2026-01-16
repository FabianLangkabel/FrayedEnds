#include "integrals_open_shell.hpp"
#include <utility> 

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
            old_akal = core_as_integrals_two_body[5];
        }
    }


  

  inline int spin_combination_to_idx(const std::array<int, 2>& spin_combination) const noexcept {
    if (spin_combination[0] == 0 && spin_combination[1] == 0) return 0;
    if (spin_combination[0] == 1 && spin_combination[1] == 1) return 1;
    if (spin_combination[0] == 0 && spin_combination[1] == 1) return 2;
    if (spin_combination[0] == 1 && spin_combination[1] == 0) return 3;
    return -1; //Error
  }

  // Integrals relevant for energy calculation
  
  inline double kl(int k, int l, int spin_combination) const noexcept {
    return as_integrals_one_body[spin_combination](k, l);
  }

  inline double phys_klmn(int k, int l, int m, int n, int spin_combination) const noexcept {
    return as_integrals_two_body[spin_combination](k, l, m, n);
  }

  inline double phys_akal(int a, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_akal[spin_combination_to_idx(spin_combination)](a, k, l);
  }

  inline double phys_akla(int a, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_akla[spin_combination_to_idx(spin_combination)](a, k, l);
  }

  // Mapping: Integrals for AS orbital refinement to stored integrals
  // phys_as_z... -> Physical notation z is AS orbital
  // phys_core_z... -> Physical notation z is core orbital

  inline double ak(int a, int k, int spin_combination) const noexcept {
    return core_as_integrals_one_body_ak[spin_combination](a, k);
  }

  inline double phys_as_zlkn(int z, int l, int k, int n, const std::array<int, 2>& spin_combination) const noexcept {
    return as_integrals_two_body[spin_combination_to_idx(spin_combination)](z, l, k, n);
  }
  inline double phys_core_zlkn(int z, int l, int k, int n, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_akln[spin_combination_to_idx(spin_combination)](z, l, k, n);
  }


  inline double phys_as_zaka(int z, int a, int k, const std::array<int, 2>& spin_combination) const noexcept {
    //switch bra and ket
    std::array<int, 2> swapped = spin_combination;
    std::swap(swapped[0], swapped[1]);
    return core_as_integrals_two_body_akal[spin_combination_to_idx(swapped)](a, z, k);
  }
  inline double phys_core_zaka(int z, int a, int k, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_abak[spin_combination_to_idx(spin_combination)](a, z, k);
  }
  inline double phys_as_zkaa(int z, int a, int k, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_akla[spin_combination_to_idx(spin_combination)](a, z, k);
  }
  inline double phys_core_zkaa(int z, int a, int k, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_baak[spin_combination_to_idx(spin_combination)](a, z, k);
  }

  // Mapping: Integrals for core orbital refinement to stored integrals
  // phys_as_z... -> Physical notation z is AS orbital
  // phys_core_z... -> Physical notation z is core orbital

  inline double phys_as_zaca(int z, int a, int c, const std::array<int, 2>& spin_combination) const noexcept {
    //Input: <za|ca> = (zc|aa); stored: <ab|ak> = (aa|bk) -> map: Alpha/Beta - Bra/KET Vertauschung lösen!
    //core_as_integrals_two_body_abak
    return 0;
  }
  inline double phys_core_zaca(int z, int a, int c, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_baca[spin_combination_to_idx(spin_combination)](a, z, c);
  }
  inline double phys_as_zaac(int z, int a, int c, const std::array<int, 2>& spin_combination) const noexcept {
    //Input: <za|ac> = (za|ac); stored: <ba|ak> = (ba|ak) -> map: Alpha/Beta - Bra/KET Vertauschung lösen!
    //core_as_integrals_two_body_baak
    return 0;
  }
  inline double phys_core_zaac(int z, int a, int c, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_baac[spin_combination_to_idx(spin_combination)](a, z, c);
  }


  inline double phys_as_zkcl(int z, int c, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    //Input: <zk|cl> = (zc|kl); stored: <ak|ln> = (al|kn) -> map: z->l, c->a, k->k, l->n
    return core_as_integrals_two_body_akln[spin_combination_to_idx(spin_combination)](c, k, z, l);
  }
  inline double phys_core_zkcl(int z, int c, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    return core_as_integrals_two_body_akcl[spin_combination_to_idx(spin_combination)](z, k, c, l);
  }
  inline double phys_as_zklc(int z, int c, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    //Input: <zk|lc> = (zl|kc); stored: <ak|ln> = (al|kn) -> map: Alpha/Beta - Bra/KET Vertauschung lösen!
    //core_as_integrals_two_body_akln
    return 0;
  }
  inline double phys_core_zklc(int z, int c, int k, int l, const std::array<int, 2>& spin_combination) const noexcept {
    core_as_integrals_two_body_aklc[spin_combination_to_idx(spin_combination)](z, k, l, c);
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

    std::vector<madness::Tensor<double>> old_akal; // (a,k,l)

    // Additional core - AS integrals for core orbital refinement
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baca; // (a,b,c)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baac; // (a,b,c)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akcl; // (a,k,c,l)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_aklc; // (a,k,l,c)
};