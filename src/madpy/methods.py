import numpy
from ._madpy_impl import SavedFct2D, SavedFct3D
from .optimization import Optimization2D, Optimization3D
from .integrals import Integrals2D, Integrals3D
from .madpno import MadPNO
from .pyscf_interface import PySCFInterface, HAS_PYSCF
from .pyscf_interface import SUPPORTED_RDM_METHODS as PYSCF_METHODS
from .tequila_interface import TequilaInterface, HAS_TEQUILA
from .tequila_interface import SUPPORTED_RDM_METHODS as TEQUILA_METHODS
from .madworld import MadWorld2D, MadWorld3D
from .madmolecule import MadMolecule
from .minbas import AtomicBasisProjector
from .eigensolver import Eigensolver2D, Eigensolver3D

SUPPORTED_RDM_METHODS = TEQUILA_METHODS + PYSCF_METHODS
AVAILABLE_RDM_METHODS = []

if HAS_TEQUILA:
    AVAILABLE_RDM_METHODS += TEQUILA_METHODS
if HAS_PYSCF:
    AVAILABLE_RDM_METHODS += PYSCF_METHODS

def optimize_basis_3D(world:MadWorld3D,
                   Vnuc: SavedFct3D = None,
                   n_electrons: int = None,
                   nuclear_repulsion = 0.0,
                   geometry=None,
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

    if Vnuc==None and geometry==None:
        raise Exception("Please provide either a potential or a molecular geometry.")
    elif Vnuc!=None:
        c = nuclear_repulsion
        if n_electrons==None:
            raise Exception("If you provide a potential, you need to specifiy the number of electrons (n_electrons).")
        if n_orbitals==None:
            n_orbitals = n_electrons #as of right now there is no frozen core implemented for calculations with a custom potential
    else:
        mol = MadMolecule(geometry)
        c = mol.get_nuclear_repulsion()
        Vnuc = mol.get_vnuc(world)
        if n_orbitals is None:
            n_orbitals = mol.n_core_electrons//2 + (mol.n_electrons - mol.n_core_electrons)

    if orbitals is None or "pno" in orbitals:
        if geometry is None:
            raise Exception("If you want to use PNOs, you need to provide a molecular geometry.")
        madpno = MadPNO(world, geometry, n_orbitals=n_orbitals)
        if many_body_method == "spa" and "edges" not in kwargs:
            kwargs["edges"] = madpno.get_spa_edges()
        orbitals = madpno.get_orbitals()
        del madpno
    elif "sto" in orbitals and "3g" in orbitals:
        if geometry is None:
            raise Exception("If you want to use the sto-3g basis, you need to provide a molecular geometry.")
        minbas = AtomicBasisProjector(world, geometry, aobasis="sto-3g")
        orbitals = minbas.orbitals
        for x in orbitals: x.type="active"
        # test if we have frozen core: if yes, we need the HF orbitals as core orbitals
        if mol.n_core_electrons > 0:
            hf = minbas.solve_scf()
            core = [hf[k] for k in range(mol.n_core_electrons//2)]
            integrals = Integrals3D(world)
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
    elif "eigen" in orbitals:
        if Vnuc is None:
            raise Exception("If you want to use the eigensolver you need to provide a potential.")
        eigen = Eigensolver3D(world, Vnuc)
        orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=n_orbitals*2)
        del eigen




    current = 0.0
    for iteration in range(maxiter):
        integrals = Integrals3D(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        del integrals

        if many_body_method in PYSCF_METHODS:
            if geometry is None:
                mol = PySCFInterface(
                    n_electrons=n_electrons, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
                )
            else:
                mol = PySCFInterface(
                    geometry=geometry, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
                )
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, return_energy=True)
        elif many_body_method in TEQUILA_METHODS:
            if geometry is None:
                mol = TequilaInterface(
                    n_electrons=n_electrons, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
                )
            else:
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

        if numpy.isclose(energy, current, atol=econv, rtol=0.0): # Formula for this is absolute(energy - current) <= (atol + rtol * absolute(current)), I feel like its more intuitive with rtol=0.0 but we can also change it back
            break
        current = energy

        if dconv is None:
            dconv = 10*econv
        if occ_thresh is None:
            occ_thresh = econv
        opti = Optimization3D(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=dconv, occ_thresh=occ_thresh
        )
        del opti

    return energy, orbitals, rdm1, rdm2

PYSCF_METHODS_2D=["fci"] #what else can be used in 2d?
TEQUILA_METHODS_2D=[]
SUPPORTED_RDM_METHODS_2D = TEQUILA_METHODS_2D + PYSCF_METHODS_2D

def optimize_basis_2D(world:MadWorld2D,
                   Vnuc: SavedFct2D,
                   n_electrons: int,
                   nuclear_repulsion = 0.0,
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

    c = nuclear_repulsion
    if n_orbitals==None:
        n_orbitals = n_electrons #as of right now there is no frozen core implemented for calculations with a custom potential
    

    if orbitals is None or "eigen" in orbitals:
        eigen = Eigensolver2D(world, Vnuc)
        orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=n_orbitals*2)
        del eigen




    current = 0.0
    for iteration in range(maxiter):
        integrals = Integrals2D(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        del integrals

        if many_body_method in PYSCF_METHODS_2D:
            mol = PySCFInterface(n_electrons=n_electrons, one_body_integrals=T + V, two_body_integrals=G, constant_term=c)
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, return_energy=True)
        elif many_body_method in TEQUILA_METHODS_2D:
            mol = TequilaInterface(n_electrons=n_electrons, one_body_integrals=T + V, two_body_integrals=G, constant_term=c)
            rdm1, rdm2, energy = mol.compute_rdms(method=many_body_method, *args, **kwargs)
        elif many_body_method == "dmrg":
            raise Exception("not here yet")
        elif callable(many_body_method):
            rdm1, rdm2, energy = many_body_method(T,V,G,c,*args, **kwargs)
        else:
            raise Exception(f"many_body_method={str(many_body_method)} is neither a string that encodes a supported method nor callable\nsupported 2D methods are: {SUPPORTED_RDM_METHODS}")

        print("iteration {} energy {:+2.5f}".format(iteration, energy))

        if numpy.isclose(energy, current, atol=econv, rtol=0.0): # Formula for this is absolute(energy - current) <= (atol + rtol * absolute(current)), I feel like its more intuitive with rtol=0.0 but we can also change it back
            break
        current = energy

        if dconv is None:
            dconv = 10*econv
        if occ_thresh is None:
            occ_thresh = econv
        opti = Optimization2D(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=dconv, occ_thresh=occ_thresh
        )
        del opti

    return energy, orbitals, rdm1, rdm2