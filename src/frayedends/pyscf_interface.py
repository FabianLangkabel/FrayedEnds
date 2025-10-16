"""
Using tequila pyscf interface to mitigate maintenance burden
"""

import numpy

HAS_TEQUILA = True
HAS_PYSCF = True
TQ_PYSCF_INTERFACE_WORKING = True
try:
    import tequila as tq

    if "pyscf" not in tq.quantumchemistry.INSTALLED_QCHEMISTRY_BACKENDS:
        try:
            import pyscf
        except ImportError as E:
            HAS_PYSCF = E
        if HAS_PYSCF:
            TQ_PYSCF_INTERFACE_WORKING = False
except ImportError as E:
    HAS_TEQUILA = E
try:
    import pyscf
except ImportError:
    HAS_PYSCF = ImportError

SUPPORTED_RDM_METHODS = [
    "cisd",
    "mp2",
    "ccsd",
    "fci",
    "fci_direct_spin1",
    "fci_direct_spin0",
    "fci_dhf_slow",
    "fci_direct_nosym",
]


class PySCFInterface:

    tqmol = None

    def __init__(
        self,
        one_body_integrals,
        two_body_integrals,
        constant_term,
        n_electrons=None,
        geometry=None,
        *args,
        **kwargs,
    ):

        if not HAS_PYSCF:
            raise ImportError(
                "{}\nPySCFINterface: pyscf not installed; pip install pyscf".format(
                    str(HAS_PYSCF)
                )
            )
        if not HAS_TEQUILA:
            raise ImportError(
                "{}\nTequila not installed; pip install tequila".format(
                    str(HAS_TEQUILA)
                )
            )
        if not TQ_PYSCF_INTERFACE_WORKING:
            raise Exception("tq-pyscf interface broken :-(")

        if not isinstance(two_body_integrals, tq.quantumchemistry.NBodyTensor):
            ordering = None  # will be auto-detected
            if "ordering" in kwargs:
                ordering = kwargs["ordering"]
                kwargs.pop("ordering")
            two_body_integrals = tq.quantumchemistry.NBodyTensor(
                two_body_integrals, ordering=ordering
            )

        two_body_integrals.reorder(to="chem")

        if n_electrons == None and geometry == None:
            raise Exception(
                "Please provide either a number of electrons or a geometry."
            )
        elif geometry != None:
            mol = tq.Molecule(
                geometry=geometry,
                one_body_integrals=one_body_integrals,
                two_body_integrals=two_body_integrals,
                nuclear_repulsion=constant_term,
                *args,
                **kwargs,
            )
            self.tqmol = tq.quantumchemistry.QuantumChemistryPySCF.from_tequila(
                molecule=mol
            )
            self.n_electrons = self.tqmol.n_electrons
            self.n_orbitals = self.tqmol.n_orbitals
            self.constant_term, self.one_body_integrals, self.two_body_integrals = (
                self.tqmol.get_integrals(ordering="chem")
            )
        else:
            self.n_electrons = (
                n_electrons  # needs to be adapted to allow for frozen core calculations
            )
            self.n_orbitals = one_body_integrals.shape[0]
            self.one_body_integrals = one_body_integrals
            self.two_body_integrals = two_body_integrals
            self.constant_term = constant_term

    def compute_energy(self, method: str, *args, **kwargs):
        if "fci" not in method and self.tqmol == None:
            raise Exception(
                "For cisd, mp2 or ccsd you need to provide a molecular geometry."
            )
        if method in SUPPORTED_RDM_METHODS:
            return self.compute_rdms(
                method=method, return_energy=True, *args, **kwargs
            )[0]
        return self.tqmol.compute_energy(method=method, *args, **kwargs)

    def compute_rdms(self, method="fci", return_energy=False, *args, **kwargs):
        method = method.lower()
        if "fci" in method:
            from pyscf import fci

            c, h1, h2 = (
                self.constant_term,
                self.one_body_integrals,
                self.two_body_integrals,
            )
            if method == "fci":
                if self.n_electrons % 2 == 0:
                    solver = fci.direct_spin0.FCI()
                else:
                    solver = fci.direct_spin1.FCI()
            elif method == "fci_dhf_slow":
                solver = fci.__dict__[method].FCI()
            else:
                nofci_m = method.replace("fci_", "")
                solver = fci.__dict__[nofci_m].FCI()

            if method == "fci_dhf_slow":  # doesn't converge great
                energy, fcivec = solver.kernel(
                    h1,
                    h2.elems,
                    self.n_orbitals,
                    self.n_electrons,
                    nroots=self.n_orbitals,
                )
                if len(fcivec) == self.n_orbitals:
                    fcivec = fcivec[0]
            else:
                energy, fcivec = solver.kernel(
                    h1, h2.elems, self.n_orbitals, self.n_electrons
                )

            energy = energy + c
            rdm1, rdm2 = solver.make_rdm12(fcivec, self.n_orbitals, self.n_electrons)
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        elif "ci" in method:
            from pyscf import ci

            hf = self.tqmol._get_hf(do_not_solve=False, **kwargs)
            cisd = ci.CISD(hf)
            cisd.kernel()
            energy = cisd.e_tot
            rdm1 = cisd.make_rdm1()
            rdm2 = cisd.make_rdm2()
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        elif "cc" in method:
            from pyscf import cc

            hf = self.tqmol._get_hf(do_not_solve=False, **kwargs)
            ccsd = cc.CCSD(hf)
            ccsd.kernel()
            energy = ccsd.e_tot
            rdm1 = ccsd.make_rdm1()
            rdm2 = ccsd.make_rdm2()
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        elif "mp2" in method:
            from pyscf import mp

            hf = self.tqmol._get_hf(do_not_solve=False, **kwargs)
            mp2 = mp.MP2(hf)
            mp2.kernel()
            energy = mp2.e_tot
            rdm1 = mp2.make_rdm1()
            rdm2 = mp2.make_rdm2()
            rdm2 = numpy.swapaxes(rdm2, 1, 2)
        else:
            raise Exception(
                f"compute_rdms: method={method} not supported (yet)\nsupported are{SUPPORTED_RDM_METHODS}"
            )

        if return_energy:
            return rdm1, rdm2, energy
        return rdm1, rdm2
