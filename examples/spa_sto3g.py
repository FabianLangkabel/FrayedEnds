import time

import numpy
import tequila as tq

import madpy


def run(R):
    energies = {}
    # initialize the PNO interface
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 {}\nH 0.0 0.0 {}\nH 0.0 0.0 {}".format(
        R, 2 * R, 3 * R
    )  # geometry in Angstrom
    print(geom)

    world = madpy.MadWorld3D()

    madpno = madpy.MadPNO(world, geom, n_orbitals=4, maxrank=1)
    orbitals = madpno.get_orbitals()
    atomics = madpno.get_sto3g()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    world.plot_lines(atomics, "atomics")

    integrals = madpy.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=atomics)
    G = integrals.compute_two_body_integrals(orbitals)
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    # del integrals

    world.plot_lines(orbitals, "o-atomics")

    mol1 = tq.Molecule(
        geom,
        one_body_integrals=T + V,
        two_body_integrals=G,
        nuclear_repulsion=nuc_repulsion,
    )
    mol2 = tq.Molecule(geom, basis_set="sto-3g").use_native_orbitals()

    for k, mol in enumerate([mol2, mol1]):
        U = mol.make_ansatz(name="SPA", edges=[(0, 1), (2, 3)])
        u = numpy.eye(4)
        u[0] = [1, 1, 0, 0]
        u[1] = [-1, 1, 0, 0]
        u[2] = [0, 0, 1, 1]
        u[3] = [0, 0, -1, 1]

        opt = tq.quantumchemistry.optimize_orbitals(
            molecule=mol, circuit=U, silent=True, initial_guess=u.T
        )
        u = opt.mo_coeff
        mol = opt.molecule
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        print(result.energy)
        print(u)
        energies["SPA/sto-3g"] = result.energy

    integrals = madpy.Integrals3D(world)
    orbitals = integrals.transform(orbitals, u)

    c = nuc_repulsion
    for x in orbitals:
        x.type = "active"

    current = result.energy
    for iteration in range(3):
        world.plot_lines(orbitals, f"orbitals-iteration-{iteration}")

        integrals = madpy.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals)
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals)

        mol = tq.Molecule(
            geom, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c
        )
        fci = mol.compute_energy("fci")
        U = mol.make_ansatz(name="SPA", edges=[(0, 1), (2, 3)])
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

        print("iteration {} energy {:+2.5f}".format(iteration, result.energy))
        energies["SPA/MRA-NO[it={}]".format(iteration)] = result.energy
        energies["FCI/MRA-NO[it={},wfn=spa]".format(iteration)] = fci

        opti = madpy.Optimization3D(world, Vnuc, nuc_repulsion)
        new_orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.01, occ_thresh=0.001
        )

        integrals = madpy.Integrals3D(world)
        S = integrals.compute_overlap_integrals(orbitals, new_orbitals)
        print("overlap new and old")
        print(S)
        # permute if necessary to avoid breaking the spa ansatz
        # virtuals are sometimes flipped
        xorbitals = [x for x in orbitals]
        for i in range(S.shape[0]):
            j = numpy.argmax(S[i])
            # currently not implented
            # we would simply need to scale the function by -1.0
            # this will only affect spa if we reuse the parameters
            if S[i][j] < 0.0:
                print("\n\n--> phase detected <--\n\n")
            orbitals[i] = new_orbitals[j]

        S = integrals.compute_overlap_integrals(xorbitals, orbitals)
        print(S)

        if iteration > 0 and numpy.isclose(current, result.energy, atol=1.0e-3):
            print("converged")
            break
        current = result.energy

    del madpno
    del integrals
    del opti
    del world

    return energies


results = []
x = list(numpy.linspace(start=0.65, stop=1.2, num=5, endpoint=False))
for R in x:
    start = time.time()
    energies = run(R=R)
    end = time.time()
    print(f"took {end-start}s")
    results += [energies]
    print(results)

import numpy as np
from matplotlib import pyplot as plt

# x = list(numpy.linspace(start=0.5, stop=5.0, num=10, endpoint=False))

# x = [0.5, 1.0, 1.5]
# results=[{'SPA/sto-3g': -1.6421587233764838, 'SPA/MRA-NO[it=0]': -1.642158705801479, 'FCI/MRA-NO[it=0,wfn=spa]': np.float64(-1.6531256690039475), 'SPA/MRA-NO[it=1]': -1.7788787896699159, 'FCI/MRA-NO[it=1,wfn=spa]': np.float64(-1.7972218267634759), 'SPA/MRA-NO[it=2]': -1.775048826106838, 'FCI/MRA-NO[it=2,wfn=spa]': np.float64(-1.7972607048504239)},{'SPA/sto-3g': -1.890230461373062, 'SPA/MRA-NO[it=0]': -1.8902306574776928, 'FCI/MRA-NO[it=0,wfn=spa]': np.float64(-1.8977741930566938), 'SPA/MRA-NO[it=1]': -2.027493680246803, 'FCI/MRA-NO[it=1,wfn=spa]': np.float64(-2.0341163294000686), 'SPA/MRA-NO[it=2]': -2.0283553809014556, 'FCI/MRA-NO[it=2,wfn=spa]': np.float64(-2.034745664067941)}, {'SPA/sto-3g': -1.9798724254066364, 'SPA/MRA-NO[it=0]': -1.979872353501851, 'FCI/MRA-NO[it=0,wfn=spa]': np.float64(-1.9961354472635269), 'SPA/MRA-NO[it=1]': -2.1017979358026113, 'FCI/MRA-NO[it=1,wfn=spa]': np.float64(-2.111823988173218), 'SPA/MRA-NO[it=2]': -2.1021726237555294, 'FCI/MRA-NO[it=2,wfn=spa]': np.float64(-2.1117874097867873)}, {'SPA/sto-3g': -1.8902304613730654, 'SPA/MRA-NO[it=0]': -1.8902306574776928, 'FCI/MRA-NO[it=0,wfn=spa]': np.float64(-1.897774193056693), 'SPA/MRA-NO[it=1]': -2.0274936802468058, 'FCI/MRA-NO[it=1,wfn=spa]': np.float64(-2.0341163294000664), 'SPA/MRA-NO[it=2]': -2.0283553809014534, 'FCI/MRA-NO[it=2,wfn=spa]': np.float64(-2.03474566406794)}]

methods = list(results[0].keys())
for m in methods:
    y = list(results[k][m] for k, xx in enumerate(x))
    print(y)
    plt.plot(x, y, label=m, marker="x")
plt.legend()
plt.show()
