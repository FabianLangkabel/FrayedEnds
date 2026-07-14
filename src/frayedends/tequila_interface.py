import numpy

HAS_TEQUILA = True
try:
    import tequila as tq
except ImportError as E:
    HAS_TEQUILA = E

SUPPORTED_RDM_METHODS = [
    "spa",
    "upccd",
    "upccgd",
    "upccgsd",
    "hcb-spa",
    "hcb-upccgd",
    "hcb-upccd",
]


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
    def tq_molecule_from_integrals(
        one_body_integrals,
        two_body_integrals,
        constant_term=0.0,
        geometry=None,
        n_electrons=None,
        *args,
        **kwargs,
    ):
        if geometry is None:
            if n_electrons is None:
                raise Exception("neither geometry nor n_electrons given")
            # make dummy geometry that has enough electrons
            geometry = "".join([f"h 0.0 0.0 {float(k)}\n" for k in range(n_electrons)])

        return tq.Molecule(
            geometry=geometry,
            one_body_integrals=one_body_integrals,
            two_body_integrals=two_body_integrals,
            nuclear_repulsion=constant_term,
        )

    def compute_rdms(
        self,
        method="spa",
        optimize_orbitals=False,
        optimizer_arguments=None,
        *args,
        **kwargs,
    ):
        method = method.lower()
        if method == "spa":
            method = "hcb-spa"
        if method == "upccgd":
            method = "hcb-upccgd"
        if method == "upccd":
            method = "hcb-upccd"
        if optimizer_arguments is None:
            optimizer_arguments = {}

        use_hcb = "hcb" in method

        if use_hcb:
            U = self.mol.make_ansatz(name=method, *args, **kwargs)
            H = self.mol.make_hardcore_boson_hamiltonian()
        else:
            U = self.mol.make_ansatz(name=method, *args, **kwargs)
            H = self.mol.make_hamiltonian()

        trafo = None
        if optimize_orbitals:
            oo_options = {"silent": True}
            if "oo_options" in kwargs:
                oo_options = {**oo_options, **kwargs["oo_options"]}
            opt = tq.quantumchemistry.optimize_orbitals(molecule=self.mol, circuit=U, use_hcb=use_hcb, **oo_options)
            if "hcb" in method:
                H = opt.molecule.make_hardcore_boson_hamiltonian()
            else:
                H = opt.molecule.make_hamiltonian()
            trafo = opt.mo_coeff.T

        E = tq.ExpectationValue(H=H, U=U)
        optimizer_arguments_default = {"silent": True, "initial_values": "near_zero"}
        optimizer_arguments = {**optimizer_arguments_default, **optimizer_arguments}
        result = tq.minimize(E, **optimizer_arguments)
        rdm1, rdm2 = self.mol.compute_rdms(U=U, use_hcb=use_hcb, variables=result.variables)
        energy = result.energy

        if trafo is not None:
            raise Exception("orbital optimization not yet supported: need to re-transform the rdms")

        return rdm1, rdm2, energy

    def compute_energy(self, *args, **kwargs):
        return self.compute_rdms(*args, **kwargs)[2]
    
    def expectation_value_orthogonality_constraint(self, H, U, circuit_list, constant_list):
        E = tq.ExpectationValue(H=H, U=U)
        if (len(circuit_list) != len(constant_list)):
            raise ValueError(f"Circuit_list and constant_list have different lengths. len(circuit_list): '{len(circuit_list)}', len(constant_list): '{len(constant_list)}'")
        list_length = len(circuit_list)
        for l in range(list_length):
            if (circuit_list[l].extract_variables() == None):
                raise ValueError(f"Circuit_list contains unparametrized elements")
        U_list = []
        for i in range(0, list_length):
            U_k = U + circuit_list[i].dagger()
            P_k = 1
            for k in U_k.qubits:
                P_k*= tq.paulis.Qp(k)
            E_k = tq.ExpectationValue(H=P_k, U=U_k)
            U_list.append(constant_list[i]*E_k)
        return E + sum(U_list)
