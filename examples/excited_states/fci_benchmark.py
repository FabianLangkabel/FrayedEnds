import tequila as tq
import numpy as np

distance = np.arange(6, 0.2, -0.02).tolist() #0.01 would be better

for d in distance:
    print(-d - 2.55, -d, d, d + 2.5)

results = []

for d in distance:
    # Dissociated hydrogen chain
    geom =  ("H 0.0 0.0 " + (-d - 2.55).__str__() + "\n"
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
            "H 0.0 0.0 " + (d + 2.55).__str__() + "\n"
    )
    mol = tq.Molecule(geometry=geom, basis_set='6-31g')
    fci_dis = mol.compute_energy('fci')
    #print("FCI Dissociated H4 Energy: ", fci_dis)

    results.append({"distance": 2*d, "fci_energy": fci_dis})

with open("fci_results.dat", "w") as f:
    f.write("# distance   FCI\n")
    for res in results:
        d = res["distance"]
        E = res["fci_energy"]
        f.write(f"{d:8.3f}  {E:15.8f} \n")

print("\nErgebnisse gespeichert in fci_results.dat\n")

"""
geom = "H 0.0 0.0 -1.0\nH 0.0 0.0 -0.5\nH 0.0 0.0 0.5\nH 0.0 0.0 1.0\n"
mol = tq.Molecule(geometry=geom, basis_set='6-31g')
fci_eq = mol.compute_energy('fci')
print("FCI Equilibrium H2 Energy: ", fci_eq)
"""