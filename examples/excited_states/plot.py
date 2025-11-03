import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/truonthu/Documents/MRA-FrayedEnds/FrayedEnds/examples/excited_states/fci_results.dat"

data = np.loadtxt(file_path, comments="#")

distance = data[:, 0]
fci = data[:, 1]

# Plot
plt.figure(figsize=(8,5), dpi=300)
plt.plot(distance, fci, label="FCI")
plt.xlabel("internuclear distance (Å)")
plt.ylabel("potential energy (eV)")
plt.grid(True)
plt.legend(loc="best")
#plt.xlim(0, 5.5)
#plt.ylim(-1.2, -0.9)  # optional
plt.tight_layout()
plt.show()

