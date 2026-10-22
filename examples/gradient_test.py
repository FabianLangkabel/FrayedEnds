import numpy as np

import frayedends as fe


def pair_rdms(z):
    c, s = np.cos(z), np.sin(z)
    rdm1 = np.diag([2 * c**2, 2 * s**2])
    d_rdm1 = np.diag([-2 * np.sin(2 * z), 2 * np.sin(2 * z)])
    d2_rdm1 = np.diag([-4 * np.cos(2 * z), 4 * np.cos(2 * z)])

    rdm2 = np.zeros((2, 2, 2, 2))
    d_rdm2 = np.zeros_like(rdm2)
    d2_rdm2 = np.zeros_like(rdm2)
    rdm2[0, 0, 0, 0], rdm2[1, 1, 1, 1] = 2 * c**2, 2 * s**2
    rdm2[0, 0, 1, 1] = rdm2[1, 1, 0, 0] = 2 * c * s
    d_rdm2[0, 0, 0, 0], d_rdm2[1, 1, 1, 1] = -2 * np.sin(2 * z), 2 * np.sin(2 * z)
    d_rdm2[0, 0, 1, 1] = d_rdm2[1, 1, 0, 0] = 2 * np.cos(2 * z)
    d2_rdm2[0, 0, 0, 0], d2_rdm2[1, 1, 1, 1] = -4 * np.cos(2 * z), 4 * np.cos(2 * z)
    d2_rdm2[0, 0, 1, 1] = d2_rdm2[1, 1, 0, 0] = -4 * np.sin(2 * z)
    return rdm1, rdm2, d_rdm1, d_rdm2, d2_rdm1, d2_rdm2


def stationary_pair_rdms(h, g, c):
    pair_hamiltonian = np.array(
        [
            [2 * h[0, 0] + g[0, 0, 0, 0], 0.5 * (g[0, 0, 1, 1] + g[1, 1, 0, 0])],
            [0.5 * (g[0, 0, 1, 1] + g[1, 1, 0, 0]), 2 * h[1, 1] + g[1, 1, 1, 1]],
        ]
    )
    e, pair_states = np.linalg.eigh(pair_hamiltonian)
    coefficients = pair_states[:, 0]
    if coefficients[0] < 0:
        coefficients = -coefficients
    return e[0] + c, np.arctan2(coefficients[1], coefficients[0]), pair_rdms(np.arctan2(coefficients[1], coefficients[0]))


def make_lagrange_multipliers(h, g, rdm1, rdm2):
    lagrange = np.zeros((2, 2))
    for p in range(2):
        for i in range(2):
            lagrange[p, i] = rdm1[i, i] * h[p, i]
            for k in range(2):
                for l in range(2):
                    for n in range(2):
                        lagrange[p, i] += rdm2[k, l, i, n] * g[p, l, k, n]
    print("Lagrange multipliers:\n", lagrange)
    return lagrange

def orb_ref(world, distance):
    geometry = fe.MolecularGeometry(f"H 0 0 {-distance/2.0}\nH 0 0 {distance/2.0}", units="bohr")
    
    def bonding(x, y, z):
        ra = np.sqrt(x * x + y * y + (z + distance/2.0) ** 2)
        rb = np.sqrt(x * x + y * y + (z - distance/2.0) ** 2)
        return np.exp(-ra) + np.exp(-rb)

    def antibonding(x, y, z):
        ra = np.sqrt(x * x + y * y + (z + distance/2.0) ** 2)
        rb = np.sqrt(x * x + y * y + (z - distance/2.0) ** 2)
        return np.exp(-ra) - np.exp(-rb)

    orbitals = [
        fe.MRAFunctionFactory(world, bonding).get_function(),
        fe.MRAFunctionFactory(world, antibonding).get_function(),
    ]
    integrals = fe.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals)
    potential = geometry.get_vnuc(world)
    nuclear_repulsion = geometry.get_nuclear_repulsion()
    c, h, g = integrals.compute_effective_hamiltonian([], orbitals, potential, nuclear_repulsion)
    current = 0
    for iteration in range(10):
        energy, angle, (rdm1, rdm2, _, _, _, _) = stationary_pair_rdms(h, g, c)
        print(f"iteration {iteration} energy {energy:.6f}")
        refinement = fe.OrbitalRefinement(
            world,
            potential,
            nuclear_repulsion
        )
        _, orbitals, orbitals_converged = refinement.refine_orbitals(
            orbitals,
            rdm1,
            rdm2,
            opt_thresh=1e-5,
            occ_thresh=1e-4,
            maxiter=3,
        )

        c, h, g = refinement.get_effective_hamiltonian()

        if abs(current - energy) < 1e-4:
            break
        current = energy


    energy, angle, (rdm1, rdm2, d_rdm1, d_rdm2, d2_rdm1, d2_rdm2) = stationary_pair_rdms(h, g, c)
    lagrange = make_lagrange_multipliers(h, g, rdm1, rdm2)
    return orbitals, energy, angle, rdm1, rdm2, d_rdm1, d_rdm2, d2_rdm1, d2_rdm2, lagrange
    

def test_h2_coupled_implicit_response_along_bond(world, distance):
    geometry = fe.MolecularGeometry(f"H 0 0 {-distance/2.0}\nH 0 0 {distance/2.0}", units="bohr")
    integrals = fe.Integrals(world)

    lmd = 0.001
    orbitals2, energy, angle2, rdm1, rdm2, d_rdm1, d_rdm2, d2_rdm1, d2_rdm2, lagrange = orb_ref(world, distance+lmd)
    orbitals, energy, angle, rdm1, rdm2, d_rdm1, d_rdm2, d2_rdm1, d2_rdm2, lagrange = orb_ref(world, distance)

    orb0_l = [orbitals2[0], orbitals[0]]
    orb1_l = [orbitals2[1], orbitals[1]]

    orb0_l = integrals.transform(orb0_l, np.array([[1.0/lmd, -1.0/lmd], [-1.0/lmd, 1.0/lmd]]))
    orb1_l = integrals.transform(orb1_l, np.array([[1.0/lmd, -1.0/lmd], [-1.0/lmd, 1.0/lmd]]))

    fd_orbs = [orb0_l[0], orb1_l[0]]

    grad = geometry.compute_dR_dE(world, rdm1, orbitals)[1][2]
    
    gradients = fe.Gradients(world)
    
    response = gradients.solve_h2( #use nonlinsol?
        orbitals,
        rdm1,
        rdm2,
        lagrange,
        geometry,
        d_rdm1,
        d_rdm2,
        d2_rdm1,
        d2_rdm2,
        max_iterations=10
    )

    olap = integrals.compute_overlap_integrals(orbitals, response.orbital_derivatives)
    print("as overlap:\n", olap)

    ola4 = integrals.compute_overlap_integrals(orbitals, fd_orbs)
    print("as overlap (fd):\n", ola4)

    olap2 = integrals.compute_overlap_integrals(response.orbital_derivatives, response.orbital_derivatives)
    print("overlap/ norm:\n", olap2)

    olap3 = integrals.compute_overlap_integrals(fd_orbs, fd_orbs)
    print("fd overlap:\n", olap3)

    def x_2(x, y, z):
        return x * x

    mra_x_2 = fe.MRAFunctionFactory(world, x_2).get_function()

    h_x_2 = integrals.compute_potential_integrals(orbitals, mra_x_2)
    O_x_2 = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            O_x_2 += rdm1[i, j] * h_x_2[i, j]

    print("x^2 expectation value:", O_x_2)

    d_O1 = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            d_O1 += d_rdm1[i, j] * h_x_2[i, j] * response.parameter_derivative

    d_O2 = 0.0
    x2_orbs = integrals.multiply(orbitals, mra_x_2)
    olap_x2 = integrals.compute_overlap_integrals(response.orbital_derivatives, x2_orbs)
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            d_O2 += 2 * rdm1[i, j] * olap_x2[i, j]

    '''
    dR2_E = 0.0
    d2_V = geometry.molecular_potential_second_derivative(world, 1, 2, 2) #plot this
    a_orbs_d2_V_a_orbs = integrals.compute_potential_integrals(orbitals, d2_V)
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            dR2_E += rdm1[i, j] * a_orbs_d2_V_a_orbs[i, j]
    dR2_c = geometry.nuclear_repulsion_second_derivative(1, 1, 2, 2)

    d_V = geometry.molecular_potential_derivative(world, 1, 2)
    a_orbs_d_V_a_orbs = integrals.compute_potential_integrals(orbitals, d_V)
    fd_par_deriv = (angle2 - angle) / lmd
    print("fd comp par deriv:", fd_par_deriv - response.parameter_derivative)
    dzdR_E = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            dzdR_E +=  d_rdm1[i, j] * a_orbs_d_V_a_orbs[i, j] * fd_par_deriv

    world.line_plot("d2_V", d2_V, axis="z", datapoints=5001)
    world.line_plot("d_V", d_V, axis="z", datapoints=5001)

    whatever = integrals.compute_overlap_integrals(response.dRdphi_E, fd_orbs)
    dphidR_E = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            dphidR_E += 2 *rdm1[i, j] * whatever[i, j]

    
    cons_check = 0.0
    adads=integrals.compute_overlap_integrals(response.dRdphi_E, orbitals)
    print("adas:", adads)
    print(integrals.compute_potential_integrals(orbitals, d_V))
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            cons_check += rdm1[i, j] * adads[i, j]
    cons_check += geometry.nuclear_repulsion_derivative(1, 2)
    print("consistency check:", cons_check)
    '''

    return energy, O_x_2, d_O1, d_O2



world = fe.MadWorld(ndims=3, L=15.0, k=5, thresh=1e-4)
E_list=[]
grad_E_list=[]
hess_E_list=[]
O_list=[]
d_O_list=[]
distance_list = [1.0+0.1*i for i in range(1)]
for distance in distance_list:
    E, O_x_2, d_O1, d_O2 = test_h2_coupled_implicit_response_along_bond(world, distance)
    E_list.append(E)
    #grad_E_list.append(grad)
    #hess_E_list.append([dR2_E, dzdR_E, dphidR_E, dR2_c])
    O_list.append(O_x_2)
    d_O_list.append([d_O1, d_O2])

print("right dzdR_E")
print("distance:", distance_list)
print("energies:", E_list)
#print("gradients:", grad_E_list)
#print("hessians:", hess_E_list)
print("x^2 expectation values:", O_list)
print("x^2 expectation value derivatives:", d_O_list)
fe.cleanup(globals())
