"""
Benzene: MRA-DMRG orbital refinement of the pi space, starting from MRA Hartree-Fock orbitals

1. NWChem SCF in a Gaussian basis
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

import frayedends as fe

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
n_threads = 8  # threads for MADNESS and block2

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

basis = "cc-pvdz"  # Gaussian basis of the NWChem calculation (initial guess for MRA-HF and source of the pi* orbitals)
nwchem_command = "nwchem"  # e.g. "mpirun -np 8 nwchem"

box_size = 50.0  # half box length L (bohr)
wavelet_order = 7
madness_thresh = 1.0e-6
eprec = 1.0e-6  # smoothing of the nuclear potential

hf_econv = 1.0e-6
hf_maxiter = 30

n_pi_orbitals = 6  # one p orbital perpendicular to the ring per carbon atom -> 3 pi + 3 pi*

dmrg_bond_dim = 100  # exact for 6 orbitals
dmrg_sweeps = 10
max_macro_iterations = 20
energy_conv = 1.0e-5  # convergence of the DMRG energy between two macro iterations
refine_opt_thresh = 1.0e-3  # convergence threshold of the orbital refinement (largest residual)
refine_occ_thresh = 1.0e-3  # orbitals with smaller occupation are not refined

output_dir = "benzene_pi"


def save_orbitals(path, core, active):
    os.makedirs(path, exist_ok=True)
    for i, orb in enumerate(core):
        orb.save_to_file(os.path.join(path, f"core_{i}.fe"))
    for i, orb in enumerate(active):
        orb.save_to_file(os.path.join(path, f"active_{i}.fe"))


os.makedirs(output_dir, exist_ok=True)
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh, n_threads=n_threads)

# ---------------------------------------------------------------------------
# 1. NWChem SCF
# ---------------------------------------------------------------------------
nwchem_prefix = fe.run_nwchem(
    geometry, basis=basis, name="benzene", workdir=os.path.join(output_dir, "nwchem"), nwchem_command=nwchem_command
)

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
hf = fe.HartreeFock(
    world,
    molecule,
    initial_orbitals=[nwchem_mos[i] for i in occupied_indices],
    initial_orbital_energies=nwchem_energies[occupied_indices],
    econv=hf_econv,
    maxiter=hf_maxiter,
)
hf_energy = hf.solve(redirect_filename=os.path.join(output_dir, "hartree_fock.log"))
hf_orbitals = hf.get_orbitals()
hf_orbital_energies = hf.get_orbital_energies()
Vnuc = hf.get_vnuc()
nuclear_repulsion = hf.get_nuclear_repulsion()
print(f"MRA-HF energy: {hf_energy:.10f}")

# ---------------------------------------------------------------------------
# 3. Active space: pi orbitals of MRA-HF + projected pi* orbitals of NWChem
# ---------------------------------------------------------------------------
parity_occupied = fe.mirror_parity(world, hf_orbitals, molecule)
pi_occupied = [i for i, p in enumerate(parity_occupied) if p < 0.0]
core_indices = [i for i, p in enumerate(parity_occupied) if p >= 0.0]
print("\nMRA-HF orbitals (parity: +1 sigma, -1 pi)")
for i, (eps, p) in enumerate(zip(hf_orbital_energies, parity_occupied)):
    print(f"  {i:3d}  eps = {eps: .6f}  parity = {p: .4f}  {'pi' if p < 0.0 else ''}")

n_pi_virtual = n_pi_orbitals - len(pi_occupied)
nwchem_virtuals = [nwchem_mos[i] for i in virtual_indices]
parity_virtual = fe.mirror_parity(world, nwchem_virtuals, molecule)
pi_candidates = [j for j, p in enumerate(parity_virtual) if p < 0.0]
weights = fe.minimal_basis_weights(world, [nwchem_virtuals[j] for j in pi_candidates], hf.get_minimal_basis())
ranking = np.argsort(-weights)
pi_virtual = sorted(pi_candidates[r] for r in ranking[:n_pi_virtual])

print("\nNWChem pi-type virtuals (selection by minimal basis weight)")
for r, j in enumerate(pi_candidates):
    mark = "selected" if j in pi_virtual else ""
    i = virtual_indices[j]
    print(
        f"  MO {i:3d}  eps = {nwchem_energies[i]: .6f}  parity = {parity_virtual[j]: .4f}  weight = {weights[r]:.4f}  {mark}"
    )

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
c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuclear_repulsion, g_ordering="chem")

driver = DMRGDriver(scratch=os.path.join(output_dir, "dmrg_tmp"), symm_type=SymmetryTypes.SU2, n_threads=n_threads)
driver.initialize_system(n_sites=len(active), n_elec=n_active_electrons, spin=0)

energies = []
for iteration in range(max_macro_iterations):
    mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=0)
    ket = driver.get_random_mps(tag=f"GS{iteration}", bond_dim=dmrg_bond_dim, nroots=1)
    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=dmrg_sweeps,
        bond_dims=[dmrg_bond_dim],
        noises=[1e-4] * 2 + [1e-5] * 2 + [0],
        thrds=[1e-10],
        iprint=0,
    )
    rdm1 = driver.get_1pdm(ket)
    rdm2 = driver.get_2pdm(ket).transpose(0, 1, 3, 2)  # block2 -> phys ordering of OrbitalRefinement
    energies.append(energy)
    print(f"Macro iteration {iteration}: DMRG energy = {energy:.10f}  occupations = {np.round(np.diag(rdm1), 4)}")

    if iteration > 0 and abs(energies[-1] - energies[-2]) < energy_conv and converged:
        break

    opti = fe.OrbitalRefinement(world, Vnuc, nuclear_repulsion, orthonormalization_method="cd")
    core, active, converged = opti.refine_orbitals(
        orbitals=[core, active],
        rdm1=rdm1,
        rdm2=rdm2,
        opt_thresh=refine_opt_thresh,
        occ_thresh=refine_occ_thresh,
        maxiter=1,
        refine_core=True,
        redirect_filename=os.path.join(output_dir, f"refinement_{iteration}.log"),
    )
    c, h1, g2 = opti.get_effective_hamiltonian(g_ordering="chem")

save_orbitals(os.path.join(output_dir, "refined_orbitals"), core, active)

print("\nSummary")
print(f"  MRA-HF energy:              {hf_energy:.10f}")
print(f"  DMRG energy (HF orbitals):  {energies[0]:.10f}")
print(f"  DMRG energy (refined):      {energies[-1]:.10f}")
with open(os.path.join(output_dir, "energies.json"), "w") as f:
    json.dump({"hf_energy": hf_energy, "dmrg_energies": energies}, f, indent=2)

fe.cleanup(globals())
