from .optimization import Optimization
from .integrals import Integrals
from .madpno import MadPNO
from .pyscf_interface import PySCFInterface, SUPPORTED_RDM_METHODS
from .madworld import MadWorld
from .madmolecule import MadMolecule

from .pyscf_interface import HAS_TEQUILA
if HAS_TEQUILA:
    import tequila as tq
else:
    tq = None

def optimize_basis(world:MadWorld, geometry, n_orbitals=None, many_body_method="fci", orbitals=None, *args, **kwargs):
    many_body_method = many_body_method.lower()

    mol = MadMolecule(geometry)

    c = mol.get_nuclear_repulsion()
    Vnuc = mol.get_vnuc(world)
    if n_orbitals is None:
        # as many orbitals as electrons
        # we also need the core orbitals (so no frozen-core effects here)
        n_orbitals = mol.n_electrons

    if orbitals is None:
        madpno = MadPNO(world, geometry, n_orbitals=n_orbitals)
        if many_body_method == "spa" and "edges" not in kwargs:
            kwargs["edges"] = madpno.get_spa_edges()
        orbitals = madpno.get_orbitals()
        del madpno

    energy = 0.0
    for iteration in range(3):
        integrals = Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        del integrals

        if many_body_method in SUPPORTED_RDM_METHODS:
            mol = PySCFInterface(
                geometry=geometry, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
            )
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, return_energy=True)
        elif many_body_method == "spa":
            mol = tq.Molecule(geometry=geometry, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c, *args, **kwargs)
            U = mol.make_ansatz(name="HCB-SPA", *args, **kwargs)
            H = mol.make_hardcore_boson_hamiltonian()
            E = tq.ExpectationValue(H=H, U=U)
            result = tq.minimize(E, silent=True)
            rdm1, rdm2 = mol.compute_rdms(U=U, use_hcb=True, variables=result.variables)
            energy = result.energy
        elif many_body_method == "dmrg":
            raise Exception("not here yet")
        elif callable(many_body_method):
            rdm1, rdm2, energy = many_body_method(T,V,G,c,*args, **kwargs)
        else:
            raise Exception(f"many_body_method={str(many_body_method)} is neither a string nor callable")
        print("iteration {} energy {:+2.5f}".format(iteration, energy))

        opti = Optimization(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )
        del opti

    return energy, orbitals, rdm1, rdm2