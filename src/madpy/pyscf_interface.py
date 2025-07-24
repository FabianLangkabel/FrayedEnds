"""
Using tequila pyscf interface to mitigate maintenance burden
"""

import numpy

HAS_TEQUILA = True
HAS_PYSCF = True
TQ_PYSCF_INTERFACE_WORKING=True
try:
    import tequila as tq
    if "pyscf" not in tq.quantumchemistry.INSTALLED_QCHEMISTRY_BACKENDS:
        try:
            import pyscf
        except ImportError as E:
            HAS_PYSCF = E
        if HAS_PYSCF: TQ_PYSCF_INTERFACE_WORKING = False
except ImportError as E:
    HAS_TEQUILA = E
try:
    import pyscf
except ImportError:
    HAS_PYSCF = ImportError

SUPPORTED_RDM_METHODS = ["fci", "cisd", "qcisd"]

class PySCFInterface:

    def __init__(self, geometry, one_body_integrals, two_body_integrals, constant_term,  *args, **kwargs):

        if not HAS_PYSCF:
            raise ImportError("{}\nPySCFINterface: pyscf not installed; pip install pyscf".format(str(HAS_PYSCF)))
        if not HAS_TEQUILA:
            raise ImportError("{}\nTequila not installed; pip install tequila".format(str(HAS_TEQUILA)))
        if not TQ_PYSCF_INTERFACE_WORKING:
            raise Exception("tq-pyscf interface broken :-(")

        mol = tq.Molecule(geometry=geometry, one_body_integrals=one_body_integrals, two_body_integrals=two_body_integrals, nuclear_repulsion=constant_term, *args, **kwargs)
        self.tqmol = tq.quantumchemistry.QuantumChemistryPySCF.from_tequila(molecule=mol)
    def compute_energy(self, method:str, *args, **kwargs):
        if method in SUPPORTED_RDM_METHODS:
            return self.compute_rdms(method=method, *args, **kwargs)[0]
        return self.tqmol.compute_energy(method=method, *args, **kwargs)

    def compute_rdms(self, method="fci", return_energy=False, *args, **kwargs):
        if method in ["fci","FCI"]:
            c, h1, h2 = self.tqmol.get_integrals(ordering="chem")
            energy, fcivec = pyscf.fci.direct_spin0.kernel(h1, h2.elems, self.tqmol.n_orbitals, self.tqmol.n_electrons)
            rdm1, rdm2 = pyscf.fci.direct_spin0.make_rdm12(fcivec, self.tqmol.n_orbitals, self.tqmol.n_electrons)
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        elif hasattr(method, "lower") and method.lower() == "cisd":
            from pyscf import ci
            hf = self.tqmol._get_hf(**kwargs)
            cisd = ci.CISD(hf)
            energy, civec = cisd.kernel()
            rdm1 = cisd.make_rdm1()
            rdm2 = cisd.make_rdm2()
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        else:
            raise Exception(f"compute_rdms: method={method} not supported (yet)\nsupported are{SUPPORTED_RDM_METHODS}")

        if return_energy:
            return rdm1, rdm2, energy
        return rdm1, rdm2



