"""
Hilfsmodul: alle MOs aus einer Molden-Datei als 2D-Schnitte (z=z_above) plotten.
Wird von rhf_s0.py und uhf_t1.py genutzt nach dem NWChem-Lauf.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pyscf.tools import molden
from pyscf.dft import numint

BOHR = 0.5291772


def plot_all_mos(molden_file, outdir, z_above=1.0, extent=4.5, ngrid=80):
    """
    Plottet alle MOs aus molden_file als 2D-Schnitt bei z = z_above (Angstrom)
    in der xy-Ebene. Speichert PNGs in outdir/ (RHF) bzw. outdir/alpha,
    outdir/beta (UHF).
    """
    result = molden.load(molden_file)
    mol, mo_energy, mo_coeff, mo_occ = result[0], result[1], result[2], result[3]

    # UHF -> (alpha, beta)-Tupel, sonst single set
    is_uhf = (isinstance(mo_coeff, (list, tuple))
              and len(mo_coeff) == 2
              and hasattr(mo_coeff[0], 'shape'))
    if is_uhf:
        sets = [
            ('alpha', np.asarray(mo_coeff[0]),
             np.asarray(mo_energy[0]), np.asarray(mo_occ[0])),
            ('beta',  np.asarray(mo_coeff[1]),
             np.asarray(mo_energy[1]), np.asarray(mo_occ[1])),
        ]
    else:
        sets = [('mo', np.asarray(mo_coeff),
                 np.asarray(mo_energy), np.asarray(mo_occ))]

    z_b = z_above / BOHR
    e_b = extent / BOHR
    xs = np.linspace(-e_b, e_b, ngrid)
    X, Y = np.meshgrid(xs, xs)
    coords = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_b)])
    ao = numint.eval_ao(mol, coords)
    atom_xy = mol.atom_coords()[:, :2] * BOHR
    atom_syms = [mol.atom_symbol(i) for i in range(mol.natm)]

    os.makedirs(outdir, exist_ok=True)
    total = 0
    for label, coeff, energies, occs in sets:
        sub = os.path.join(outdir, label) if is_uhf else outdir
        os.makedirs(sub, exist_ok=True)
        nmo = coeff.shape[1]
        for i in range(nmo):
            psi = (ao @ coeff[:, i]).reshape(ngrid, ngrid)
            vmax = max(abs(psi).max(), 1e-12)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.contourf(X * BOHR, Y * BOHR, psi, levels=21,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            for xy, sym in zip(atom_xy, atom_syms):
                if sym == 'C':
                    ax.plot(xy[0], xy[1], 'o', color='black', markersize=7,
                            markerfacecolor='white', markeredgewidth=1.2)
            ax.set_aspect('equal')
            ax.set_title(
                f"{label} MO {i+1}  E={energies[i]:+.4f}  occ={occs[i]:.2f}",
                fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(f"{sub}/mo_{i+1:03d}.png",
                        dpi=90, bbox_inches='tight')
            plt.close(fig)
        print(f"  {nmo} {label}-Orbitalplots -> {sub}/")
        total += nmo
    print(f"  Gesamt: {total} PNGs in {outdir}/")


def find_molden_file(directory="."):
    """Findet die juengste .molden Datei in directory."""
    candidates = glob.glob(os.path.join(directory, "*.molden"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]
