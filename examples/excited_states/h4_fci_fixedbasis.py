import tequila as tq
import numpy as np

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":
    distance = np.arange(2.5, 0.45, -0.05).tolist()  # for linear H4 molecule with equidistant spacing d
elif geometry_mode == "h2_pair":
    distance = np.arange(1.5, 0.25, -0.02).tolist() # for H2 pair getting closer
else:
    raise ValueError("Invalid geometry mode selected.")

results = []

for d in distance:
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d
    print(f"Distance: {reported_distance: .3f}")

    if geometry_mode == "equidistant": # for equidistant linear H4 molecule
        geom = ("H 0.0 0.0 " + (-d-d/2).__str__() + "\n"
                "H 0.0 0.0 " + (-d/2).__str__() + "\n"
                "H 0.0 0.0 " + (d/2).__str__() + "\n"
                "H 0.0 0.0 " + (d+d/2).__str__() + "\n"
        )
    elif geometry_mode == "h2_pair": # for H2 molecules getting closer and closer to a H4 molecule
        geom = ("H 0.0 0.0 " + (-d - 1.5).__str__() + "\n"
                "H 0.0 0.0 " + (-d).__str__() + "\n"
                "H 0.0 0.0 " + d.__str__() + "\n"
                "H 0.0 0.0 " + (d + 1.5).__str__() + "\n"
                )
    else:
        raise ValueError("Invalid geometry mode selected.")

    mol = tq.Molecule(geometry=geom, basis_set='6-31g')
    fci_energy = mol.compute_energy('fci')

    results.append({"distance": reported_distance, "energy_0": fci_energy})

with open("results_fci_fb.dat", "w") as f:
    f.write("distance energy_0\n")
    for res in results:
        d = res["distance"]
        E = res["energy_0"]
        f.write(f"{d:8.3f}  {E:15.8f} \n")
