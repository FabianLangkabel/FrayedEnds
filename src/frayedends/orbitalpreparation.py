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

    # norm of Q|v> for Q = 1 - sum_i |i><i|: <v|v> - 2 sum_i <v|i><i|v> + sum_ij <v|i><i|j><j|v>
    S_ov = integrals.compute_overlap_integrals(occupied, virtuals)
    S_oo = integrals.compute_overlap_integrals(occupied)
    projected_norms_sq = 1.0 - 2.0 * np.einsum("iv,iv->v", S_ov, S_ov) + np.einsum("iv,ij,jv->v", S_ov, S_oo, S_ov)
    projected_norms = np.sqrt(np.clip(projected_norms_sq, 0.0, None))

    kept = [i for i, norm in enumerate(projected_norms) if norm > norm_thresh]
    info = {"kept": kept, "projected_norms": projected_norms, "overlap_with_input": np.array([])}
    if len(kept) == 0:
        return [], info

    # project_out normalizes the projected functions
    projected = integrals.project_out(occupied, [virtuals[i] for i in kept])

    S = integrals.compute_overlap_integrals(projected)
    eigenvalues = np.linalg.eigvalsh(S)
    if eigenvalues[0] < lindep_thresh * eigenvalues[-1]:
        raise ValueError(
            f"The projected virtuals are (nearly) linearly dependent (smallest/largest eigenvalue of the overlap "
            f"matrix: {eigenvalues[0] / eigenvalues[-1]:.3e}). Select fewer virtuals."
        )
    projected = integrals.orthonormalize(projected, method="symmetric")

    info["overlap_with_input"] = np.diag(integrals.compute_overlap_integrals([virtuals[i] for i in kept], projected))
    return projected, info


def mirror_parity(madworld, orbitals, molecule, offsets=(0.7, 1.4), in_plane_shift=0.7, plane_tol=1.0e-2):
    """
    Symmetry of orbitals with respect to the reflection through the plane of a planar molecule.
    For every orbital the parity
        p = sum_r phi(r) phi(r') / sum_r (phi(r)^2 + phi(r')^2) / 2
    is evaluated on sample points r above the molecular plane (around every atom) and their mirror images r'.
    p ~ +1: symmetric (sigma-type orbital), p ~ -1: antisymmetric (pi-type orbital)

    offsets: distances (bohr) of the sample points from the molecular plane
    in_plane_shift: shift (bohr) of the sample points from the atom positions within the plane
    plane_tol: maximal distance (bohr) of an atom from the fitted molecular plane
    """
    coords = np.array(molecule.to_json()["geometry"])  # bohr
    center = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - center)
    u, v, normal = vt[0], vt[1], vt[2]
    deviation = np.abs((coords - center) @ normal).max()
    if deviation > plane_tol:
        raise ValueError(f"The molecule is not planar (max. distance of an atom from the plane: {deviation:.3e} bohr).")

    above = []
    below = []
    for atom in coords:
        for du, dv in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            base = atom + in_plane_shift * (du * u + dv * v)
            for h in offsets:
                above.append(base + h * normal)
                below.append(base - h * normal)
    n = len(above)
    points = np.concatenate([np.ravel(above), np.ravel(below)]).tolist()

    values = np.array(madworld.evaluate(list(orbitals), points, units="bohr"))
    values_above = values[:, :n]
    values_below = values[:, n:]
    return np.sum(values_above * values_below, axis=1) / (0.5 * np.sum(values_above**2 + values_below**2, axis=1))


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
