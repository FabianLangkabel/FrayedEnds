import numpy as np
import matplotlib.pyplot as plt

file_path = "results_fci_fb.dat"

data =np.genfromtxt(file_path, names=True)

distance = data["distance"]
fci = data["FCI"]

# Plot
plt.figure(figsize=(8,5), dpi=300)
plt.plot(distance, fci, label="FCI_fixed")
plt.xlabel("internuclear distance (Å)")
plt.ylabel("potential energy (eV)")
plt.grid(True)
plt.legend(loc="best")
#plt.xlim(0, 5.5)
#plt.ylim(-2.3, -1.8)  # optional
plt.tight_layout()
plt.show()

