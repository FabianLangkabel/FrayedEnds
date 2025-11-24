import tequila as tq
import numpy as np

# distance = np.arange(1.5, 0.2, -0.03).tolist() for H2 pair getting closer
distance = np.arange(2.5, 0.45, -0.05).tolist()

results = []

for d in distance:
    # Dissociated hydrogen chain
    geom =  ("H 0.0 0.0 " + (-d-d/2).__str__() + "\n"
            "H 0.0 0.0 " + (-d/2).__str__() + "\n"
            "H 0.0 0.0 " + (d/2).__str__() + "\n"
            "H 0.0 0.0 " + (d+d/2).__str__() + "\n"
    )
    ''' for H2 molecules getting closer and closer to a H4 molecule
    geom = ("H 0.0 0.0 " + (-d - 2.55).__str__() + "\n"
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
            "H 0.0 0.0 " + (d + 2.55).__str__() + "\n"
            )
    '''
    mol = tq.Molecule(geometry=geom, basis_set='6-31g')
    fci_dis = mol.compute_energy('fci')
    #print("FCI Dissociated H4 Energy: ", fci_dis)

    results.append({"distance": d, "fci_energy": fci_dis}) # for H2 pair use 2*d

with open("results_fci_fb.dat", "w") as f:
    f.write("# distance   FCI\n")
    for res in results:
        d = res["distance"]
        E = res["fci_energy"]
        f.write(f"{d:8.3f}  {E:15.8f} \n")
