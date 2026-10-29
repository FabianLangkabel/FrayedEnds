import numpy as np

from .integrals import Integrals


def project_virtual_orbitals(madworld, occupied, virtuals, norm_thresh=1.0e-6, lindep_thresh=1.0e-8):
    """
    Translates virtual orbitals (e.g. NWChem virtuals in MRA representation) into the virtual space
    of a set of orthonormal occupied orbitals (e.g. MRA Hartree-Fock orbitals):
    1. normalize the virtuals
    2. project out the occupied space: |v'> = (1 - sum_i |i><i|) |v>
    3. discard virtuals with a projected norm <= norm_thresh
    4. normalize and orthonormalize symmetrically (Loewdin), which keeps the orbitals as close as possible
       to the original virtuals

    returns the projected virtuals and a dictionary with diagnostics:
        kept               indices of the input virtuals that were kept
        projected_norms    norms of all input virtuals after step 2
        overlap_with_input <v_i|v'_i> of the kept virtuals before and after the projection
    """
    integrals = Integrals(madworld)
    occupied = list(occupied)
    virtuals = integrals.normalize(list(virtuals))

    projected = integrals.project_out(occupied, virtuals, normalize=False)
    projected_norms = np.sqrt(np.clip(np.diag(integrals.compute_overlap_integrals(projected)), 0.0, None))

    kept = [i for i, norm in enumerate(projected_norms) if norm > norm_thresh]
    info = {"kept": kept, "projected_norms": projected_norms, "overlap_with_input": np.array([])}
    if len(kept) == 0:
        return [], info

    projected = integrals.normalize([projected[i] for i in kept])

    S = integrals.compute_overlap_integrals(projected)
    eigenvalues = np.linalg.eigvalsh(S)
    if eigenvalues[0] < lindep_thresh:
        raise ValueError(
            f"The projected virtuals are (nearly) linearly dependent smallest eigenvalue of the overlap "
            f"matrix: {eigenvalues[0]:.3e}). Select fewer virtuals."
        )
    projected = integrals.orthonormalize(projected, method="symmetric")

    info["overlap_with_input"] = np.diag(integrals.compute_overlap_integrals([virtuals[i] for i in kept], projected))
    return projected, info


def mirror_parity(madworld, orbitals, molecule, plane_tol=1.0e-2):
    """
    Symmetry of orbitals with respect to the reflection R through the plane of a planar molecule:
        p = <phi|R|phi> / <phi|phi>
    p = +1: symmetric (sigma-type orbital), p = -1: antisymmetric (pi-type orbital),
    values in between measure the mixing: p = w_sym - w_antisym for the weights of the (anti)symmetric parts.
    The molecule has to lie in a coordinate plane (xy, xz or yz) through the origin.

    plane_tol: maximal distance (bohr) of an atom from that plane
    """
    coords = np.array(molecule.to_json()["geometry"])  # bohr
    axes = np.flatnonzero(np.abs(coords).max(axis=0) <= plane_tol)
    if len(axes) == 0:
        raise ValueError("The molecule has to lie in a coordinate plane (xy, xz or yz) through the origin.")
    mirrormap = [1] * coords.shape[1]
    mirrormap[axes[0]] = -1

    integrals = Integrals(madworld)
    orbitals = integrals.normalize(orbitals)
    return np.diag(integrals.compute_overlap_integrals(orbitals, integrals.mirror(orbitals, mirrormap)))


def minimal_basis_weights(madworld, orbitals, minimal_basis):
    """
    Weight of every orbital in the space spanned by a minimal AO basis (e.g. HartreeFock.get_minimal_basis()):
        w_i = <phi_i|P|phi_i> / <phi_i|phi_i>,  P: projector onto the span of the minimal basis
    Valence orbitals have large weights, diffuse (Rydberg-like) orbitals small weights.
    """
    integrals = Integrals(madworld)
    S_aa = integrals.compute_overlap_integrals(minimal_basis)
    S_ao = integrals.compute_overlap_integrals(minimal_basis, orbitals)
    S_oo = integrals.compute_overlap_integrals(orbitals)
    S_aa_inv = np.linalg.pinv(S_aa, rcond=1.0e-10, hermitian=True)
    return np.einsum("ai,ab,bi->i", S_ao, S_aa_inv, S_ao) / np.diag(S_oo)
