import numpy
from .madworld import redirect_output
from .pyscf_interface import PySCFInterface

HAS_BLOCK2=True
SUPPORTED_RDM_METHDOS=["dmrg"]
try:
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except ImportError as Error:
    HAS_BLOCK2 = False

class Block2Interface(PySCFInterface):

    @redirect_output("block2.out")
    def compute_rdms(self, method="dmrg", return_energy=False, spin=None, *args, **kwargs):

        if spin is None:
            if self.tqmol.n_electrons %2 ==0:
                spin=0
            else:
                raise Exception(f"can't auto-detect spin for {self.tqmol.n_electrons} electrons")
        mol = self.tqmol

        c, h1, g2 = self.tqmol.get_integrals(ordering="chem")
        g2 = g2.elems

        driver = DMRGDriver()
        driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=spin)
        mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[100], iprint=1)

        #### Orbital reordering
        idx = driver.orbital_reordering(h1, g2)
        h1_new = h1[idx][:, idx]
        g2_new = g2[idx][:, idx][:, :, idx][:, :, :, idx]

        #### Main DMRG calculation
        driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
        mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=c, iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[200], iprint=1)

        #### PDM extraction
        pdm1 = driver.get_1pdm(ket)
        pdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)
        print('Energy from pdms = %20.15f' % (numpy.einsum('ij,ij->', pdm1, h1_new) + 0.5 * numpy.einsum('ijkl,ijkl->', pdm2,
                                                                                                   driver.unpack_g2e(
                                                                                                       g2_new)) + c))

        idx_back = numpy.zeros(len(idx), dtype=int)
        for i in range(len(idx)):
            idx_back[idx[i]] = i

        pdm1 = pdm1[idx_back][:, idx_back]
        pdm2 = pdm2[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
        pdm2 = numpy.swapaxes(pdm2, 1, 2)  # chemistry to physics notation

        dmrg_energy = numpy.einsum('ij,ij->', pdm1, h1) + 0.5 * numpy.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(g2)) + c

        if return_energy:
            return pdm1, pdm2, dmrg_energy
        else:
            return pdm1, pdm2
