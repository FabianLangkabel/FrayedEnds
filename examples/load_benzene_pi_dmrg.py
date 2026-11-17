"""
Benzene: MRA-DMRG orbital refinement of the pi space, starting from MRA Hartree-Fock orbitals

1. NWChem SCF in a Gaussian basis, the occupied orbitals are localized (Pipek-Mezey)
2. MRA Hartree-Fock (MADNESS SCF solver) started from the NWChem orbitals
3. Active space: the occupied pi orbitals of MRA-HF and pi* virtuals of NWChem, which are projected onto
   the virtual space of MRA-HF and orthonormalized. sigma and pi orbitals are distinguished by their parity
   with respect to the molecular plane, valence pi* orbitals are selected by their weight in the minimal basis.
4. DMRG (block2) in the active space and MRA orbital refinement of the core and active orbitals,
   iterated until the energy is converged
"""

import json
import os

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from time import time

import frayedends as fe

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

geometry = """
C    1.39640000   0.00000000   0.00000000
C    0.69820000   1.20931787   0.00000000
C   -0.69820000   1.20931787   0.00000000
C   -1.39640000   0.00000000   0.00000000
C   -0.69820000  -1.20931787   0.00000000
C    0.69820000  -1.20931787   0.00000000
H    2.47950000   0.00000000   0.00000000
H    1.23975000   2.14730999   0.00000000
H   -1.23975000   2.14730999   0.00000000
H   -2.47950000   0.00000000   0.00000000
H   -1.23975000  -2.14730999   0.00000000
H    1.23975000  -2.14730999   0.00000000
"""  # angstrom


basis = "sto-3g"  # Gaussian basis of the NWChem calculation (initial guess for MRA-HF and source of the pi* orbitals)
nwchem_command = "nwchem"  # e.g. "mpirun -np 8 nwchem"
nwchem_localize = None  # localization of the occupied NWChem orbitals ("pm", "boys", "ibo" or None for canonical)

box_size = 50.0  # half box length L (bohr)
wavelet_order = 7
madness_thresh = 1.0e-6
eprec = 1.0e-6  # smoothing of the nuclear potential

hf_econv = 1.0e-6
hf_maxiter = 30

n_pi_orbitals = 6  # one p orbital perpendicular to the ring per carbon atom -> 3 pi + 3 pi*

# in every macro iteration DMRG is converged for each bond dimension in turn,
# the MPS of the previous bond dimension is the initial guess for the next one
dmrg_bond_dims = [100, 250, 500, 1000]
dmrg_sweeps = 10  # maximum number of sweeps per bond dimension
dmrg_stack_mem = 8 << 30  # block2 memory in bytes (default of 1 GiB is too small for large bond dimensions)
max_macro_iterations = 20
energy_conv = 1.0e-5  # convergence of the DMRG energy between two macro iterations
refine_opt_thresh = 1.0e-3  # convergence threshold of the orbital refinement (largest residual)
refine_occ_thresh = 1.0e-3  # orbitals with smaller occupation are not refined

output_dir = "benzene_pi_load"
input_dir = "/Users/timo/workspace/MRA_nanobind/FrayedEnds/examples/benzene_pi_canon"


def save_orbitals(path, core, active):
    os.makedirs(path, exist_ok=True)
    for i, orb in enumerate(core):
        orb.save_to_file(os.path.join(path, f"core_{i}.fe"))
    for i, orb in enumerate(active):
        orb.save_to_file(os.path.join(path, f"active_{i}.fe"))

def load_orbitals(path, name):
    result = []
    didntbreakyet = True
    i = 0
    while didntbreakyet:
        filename = os.path.join(path, f"{name}{i}.fe")
        try:
            result.append(fe.SavedFct3D(filename))
            i += 1
        except:
            didntbreakyet = False
    return result

#reordering for dmrg
def reorder_active(driver, active, h1, g2, method="fiedler"):
    idx = driver.orbital_reordering(h1, g2, method=method)
    return [active[i] for i in idx], h1[np.ix_(idx, idx)], g2[np.ix_(idx, idx, idx, idx)]


os.makedirs(output_dir, exist_ok=True)
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

# ---------------------------------------------------------------------------
# 1. NWChem SCF
# ---------------------------------------------------------------------------
nwchem_prefix = os.path.join(input_dir, "nwchem/benzene")

converter = fe.NWChem_Converter(world)
converter.read_nwchem_file(nwchem_prefix)
nwchem_mos = converter.get_mos()
nwchem_occupancies = np.array(converter.get_occupancies())
nwchem_energies = np.array(converter.get_orbital_energies())
# the molecule is taken from the NWChem output to guarantee the same coordinate frame for all orbitals
molecule = converter.get_molecular_geometry(eprec=eprec)
del converter  # holds a second copy of all AOs and MOs


occupied_indices = [i for i, occ in enumerate(nwchem_occupancies) if occ > 0.0]
virtual_indices = [i for i, occ in enumerate(nwchem_occupancies) if occ == 0.0]
print(f"NWChem: {len(occupied_indices)} occupied and {len(virtual_indices)} virtual orbitals")

# ---------------------------------------------------------------------------
# 2. MRA Hartree-Fock, started from the occupied NWChem orbitals
# ---------------------------------------------------------------------------

hf_orbitals = load_orbitals(os.path.join(input_dir, "hf_nw_orbs"), "core_")
hf_energy = -230.7972069424
Vnuc = molecule.get_vnuc(world)
nuclear_repulsion = molecule.get_nuclear_repulsion()


#save_orbitals(os.path.join(output_dir, "hf_nw_orbs"), hf_orbitals, nwchem_mos)


# ---------------------------------------------------------------------------
# 3. Active space: pi orbitals of MRA-HF + projected pi* orbitals of NWChem
# ---------------------------------------------------------------------------
parity_occupied = fe.mirror_parity(world, hf_orbitals, molecule)
pi_occupied = [i for i, p in enumerate(parity_occupied) if p < 0.0]
core_indices = [i for i, p in enumerate(parity_occupied) if p >= 0.0]
print("\nMRA-HF orbitals (parity: +1 sigma, -1 pi)")
for i, p in enumerate(parity_occupied):
    print(f"  {i:3d}  parity = {p: .4f}  {'pi' if p < 0.0 else ''}")

n_pi_virtual = n_pi_orbitals - len(pi_occupied)
nwchem_virtuals = [nwchem_mos[i] for i in virtual_indices]
parity_virtual = fe.mirror_parity(world, nwchem_virtuals, molecule)
pi_virtual = [j for j, p in enumerate(parity_virtual) if p < 0.0]


pi_star, projection_info = fe.project_virtual_orbitals(world, hf_orbitals, [nwchem_virtuals[j] for j in pi_virtual])
if len(pi_star) != n_pi_virtual:
    raise RuntimeError("Projection onto the MRA-HF virtual space removed pi* orbitals.")
print("\nProjected pi* orbitals")
for j, norm, overlap in zip(pi_virtual, projection_info["projected_norms"], projection_info["overlap_with_input"]):
    print(
        f"  MO {virtual_indices[j]:3d}  norm after projection = {norm:.6f}  overlap with NWChem orbital = {overlap:.6f}"
    )

core = [hf_orbitals[i] for i in core_indices]
active = [hf_orbitals[i] for i in pi_occupied] + pi_star
n_active_electrons = 2 * len(pi_occupied)
print(f"\nActive space: {len(active)} orbitals, {n_active_electrons} electrons, {len(core)} frozen core orbitals")
save_orbitals(os.path.join(output_dir, "initial_orbitals"), core, active)

# ---------------------------------------------------------------------------
# 4. DMRG + MRA orbital refinement
# ---------------------------------------------------------------------------
integrals = fe.Integrals(world)
start_time = time()
c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuclear_repulsion, g_ordering="chem")
print(f"time integrals: {time() - start_time:.2f} s")

energies = []  # DMRG energy with the largest bond dimension of every macro iteration
bond_dim_results = []  # (bond dimension, energy, discarded weight) of every macro iteration
for iteration in range(max_macro_iterations):
    start_time = time()
    driver = DMRGDriver(
        scratch=os.path.join(output_dir, "dmrg_tmp"),
        symm_type=SymmetryTypes.SU2,
        stack_mem=dmrg_stack_mem,
    )
    driver.initialize_system(n_sites=len(active), n_elec=n_active_electrons, spin=0)

    active, h1, g2 = reorder_active(driver, active, h1, g2, method="fiedler")
    mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=0)
    ket = driver.get_random_mps(tag=f"GS{iteration}", bond_dim=dmrg_bond_dims[0], nroots=1)
    bond_dim_results.append([])
    for bond_dim in dmrg_bond_dims:  # ket is optimized in place -> initial guess for the next bond dimension
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=dmrg_sweeps,
            bond_dims=[bond_dim],
            noises=[1e-4] * 2 + [1e-5] * 2 + [0],
            thrds=[1e-10],
            iprint=0,
        )
        discarded_weight = driver.get_dmrg_results()[1][-1]
        bond_dim_results[-1].append((bond_dim, energy, discarded_weight))
        print(f"  M = {bond_dim:5d}: DMRG energy = {energy:.10f}  discarded weight = {discarded_weight:.2e}")
    rdm1 = driver.get_1pdm(ket)
    rdm2 = driver.get_2pdm(ket).transpose(0, 1, 3, 2)  # block2 -> phys ordering of OrbitalRefinement
    energies.append(energy)
    print(f"Macro iteration {iteration}: DMRG energy = {energy:.10f}  occupations = {np.round(np.diag(rdm1), 4)}")
    print(f"time dmrg: {time() - start_time:.2f} s")
    if iteration > 0 and abs(energies[-1] - energies[-2]) < energy_conv and converged:
        break

    start_time = time()
    opti = fe.OrbitalRefinement(world, Vnuc, nuclear_repulsion, orthonormalization_method="cd")
    core, active, converged = opti.refine_orbitals(
        orbitals=[core, active],
        rdm1=rdm1,
        rdm2=rdm2,
        opt_thresh=refine_opt_thresh,
        occ_thresh=refine_occ_thresh,
        maxiter=1,
        refine_core=False,
        redirect_filename=os.path.join(output_dir, f"refinement_{iteration}.log"),
    )
    c, h1, g2 = opti.get_effective_hamiltonian(g_ordering="chem")
    print(f"time orbital refinement: {time() - start_time:.2f} s")

    save_orbitals(os.path.join(output_dir, "current_orbitals"), core, active)


print("\nSummary")
print(f"  MRA-HF energy:              {hf_energy:.10f}")
print(f"  DMRG energy (HF orbitals):  {energies[0]:.10f}")
print(f"  DMRG energy (refined):      {energies[-1]:.10f}")
with open(os.path.join(output_dir, "energies.json"), "w") as f:
    json.dump({"hf_energy": hf_energy, "dmrg_energies": energies, "bond_dim_results": bond_dim_results}, f, indent=2)

fe.cleanup(globals())
