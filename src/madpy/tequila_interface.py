import numpy

HAS_TEQUILA = True
try:
    import tequila as tq
except ImportError as E:
    HAS_TEQUILA = E

SUPPORTED_RDM_METHODS=["spa", "upccd", "upccgd", "upccgsd", "hcb-spa", "hcb-upccgd", "hcb-upccd"]

class TequilaInterface:
    def __init__(self, mol=None, *args, **kwargs):
        if mol is None:
            if "one_body_integrals" in kwargs:
                mol = self.tq_molecule_from_integrals(**kwargs)
            else:
                raise Exception("neither tq molecule, nor integrals provided")
        self.mol = mol
        self.variables = 0.0

    @classmethod
    def from_molecule(cls, mol):
        return cls(mol=mol)

    @classmethod
    def from_integrals(cls, *args, **kwargs):
        return cls(mol=cls.tq_molecule_from_integrals(*args, **kwargs))
    @staticmethod
    def tq_molecule_from_integrals(one_body_integrals, two_body_integrals, constant_term=0.0, geometry=None, n_electrons=None, *args, **kwargs):
        if geometry is None:
            if n_electrons is None: raise Exception("neither geometry nor n_electrons given")
            # make dummy geometry that has enough electrons
            geometry = "".join([f"h 0.0 0.0 {float(k)}\n" for k in range(n_electrons)])

        return tq.Molecule(geometry=geometry, one_body_integrals=one_body_integrals, two_body_integrals=two_body_integrals, nuclear_repulsion=constant_term)

    def compute_rdms(self, method="spa", optimize_orbitals=False, *args, **kwargs):
        method = method.lower()
        if method == "spa": method = "hcb-spa"
        if method == "upccgd": method = "hcb-upccgd"
        if method == "upccd": method = "hcb-upccd"

        if "hcb" in method:
            U = self.mol.make_ansatz(name=method, *args, **kwargs)
            H = self.mol.make_hardcore_boson_hamiltonian()
        else:
            U = self.mol.make_ansatz(name=method, *args, **kwargs)
            H = self.mol.make_hamiltonian()

        trafo = None
        if optimize_orbitals:
            oo_options = {"initial_guess":"near_zero", "silent":True}
            if not "oo_options" in kwargs:
                oo_options = {**oo_options, **kwargs["oo_options"]}
            opt = tq.quantumchemistry.optimize_orbitals(molecule=self.mol, circuit=U, use_hcb="hcb" in method, **oo_options)
            if "hcb" in method:
                H = opt.molecule.make_hardcore_boson_hamiltonian()
            else:
                H = opt.molecule.make_hamiltonian()
            trafo = opt.mo_coeff.T

        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        rdm1, rdm2 = self.mol.compute_rdms(U=U, use_hcb=True, variables=result.variables)
        energy = result.energy

        if trafo is not None:
            pass

        return rdm1, rdm2, energy

    def compute_energy(self, *args, **kwargs):
        return self.compute_rdms(*args, **kwargs)[2]
