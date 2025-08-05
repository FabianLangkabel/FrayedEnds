import numpy

from .optimization import Optimization
from .integrals import Integrals
from .madpno import MadPNO
from .pyscf_interface import PySCFInterface, HAS_PYSCF
from .pyscf_interface import SUPPORTED_RDM_METHODS as PYSCF_METHODS
from .tequila_interface import TequilaInterface, HAS_TEQUILA
from .tequila_interface import SUPPORTED_RDM_METHODS as TEQUILA_METHODS
from .madworld import MadWorld
from .madmolecule import MadMolecule
from .minbas import AtomicBasisProjector

SUPPORTED_RDM_METHODS = TEQUILA_METHODS + PYSCF_METHODS
AVAILABLE_RDM_METHODS = []

if HAS_TEQUILA:
    AVAILABLE_RDM_METHODS += TEQUILA_METHODS
if HAS_PYSCF:
    AVAILABLE_RDM_METHODS += PYSCF_METHODS

def optimize_basis(world:MadWorld,
                   geometry,
                   n_orbitals=None,
                   many_body_method="fci",
                   orbitals=None,
                   maxiter=4,
                   econv=1.e-4,
                   dconv=None,
                   occ_thresh=None,
                   *args, **kwargs):
    many_body_method = many_body_method.lower()
    if hasattr(orbitals, "lower"): orbitals = orbitals.lower()

    mol = MadMolecule(geometry)
    c = mol.get_nuclear_repulsion()
    Vnuc = mol.get_vnuc(world)

    if n_orbitals is None:
        n_orbitals = mol.n_core_electrons//2 + (mol.n_electrons - mol.n_core_electrons)

    if orbitals is None or "pno" in orbitals:
        madpno = MadPNO(world, geometry, n_orbitals=n_orbitals)
        if many_body_method == "spa" and "edges" not in kwargs:
            kwargs["edges"] = madpno.get_spa_edges()
        orbitals = madpno.get_orbitals()
        del madpno
    elif "sto" in orbitals and "3g" in orbitals:
        minbas = AtomicBasisProjector(world, geometry, aobasis="sto-3g")
        orbitals = minbas.orbitals
        for x in orbitals: x.type="active"
        # test if we have frozen core: if yes, we need the HF orbitals as core orbitals
        if mol.n_core_electrons > 0:
            hf = minbas.solve_scf()
            core = [hf[k] for k in range(mol.n_core_electrons//2)]
            integrals = Integrals(world)
            orbitals = integrals.orthonormalize(orbitals, method="symmetric")
            orbitals = integrals.project_out(kernel=core, target=orbitals)
            orbitals = integrals.normalize(orbitals)
            # most likely no linear dependencies since core at CBS is different from sto-3g orbitals
            orbitals = integrals.orthonormalize(orbitals, method="rr_cholesky", rr_thresh=1.e-5)
            for x in core: x.type="frozen_occ"
            for x in orbitals: x.type="active"
            orbitals = core + orbitals
            # just to be save
            orbitals = integrals.normalize(orbitals)



    current = 0.0
    for iteration in range(maxiter):
        integrals = Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        del integrals

        if many_body_method in PYSCF_METHODS:
            mol = PySCFInterface(
                geometry=geometry, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
            )
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, return_energy=True)
        elif many_body_method in TEQUILA_METHODS:
            mol = TequilaInterface(
                geometry=geometry, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
            )
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, *args, **kwargs)
        elif many_body_method == "dmrg":
            raise Exception("not here yet")
        elif callable(many_body_method):
            rdm1, rdm2, energy = many_body_method(T,V,G,c,*args, **kwargs)
        else:
            raise Exception(f"many_body_method={str(many_body_method)} is neither a string that encodes a supported method nor callable\nsupported methods are: {SUPPORTED_RDM_METHODS}")

        print("iteration {} energy {:+2.5f}".format(iteration, energy))

        if numpy.isclose(energy, current, atol=econv):
            break
        current = energy

        if dconv is None:
            dconv = 10*econv
        if occ_thresh is None:
            occ_thresh = econv
        opti = Optimization(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=dconv, occ_thresh=occ_thresh
        )
        del opti

    return energy, orbitals, rdm1, rdm2