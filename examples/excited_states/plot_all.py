import numpy as np
import matplotlib.pyplot as plt

path = "/Users/truonthu/Documents/MRA-FrayedEnds/Data/linear_h4/"

fci_fb_path = path + "results_fci_fb.dat"
fci_opt_path = path + "results_fci_opt.dat"
nwchem_dmrg_oo_path = path + "results_nwchem_dmrg_oo.dat" # oo for orbital optimization
pno_dmrg_oo_path = path + "results_pno_dmrg_oo.dat"    # oo for orbital optimization
pno_fci_oo_path = path + "results_pno_fci_oo.dat"      # oo for orbital optimization


nwchem_dmrg_631g_path = path + "results_nwchem_dmrg_6-31g.dat"
nwchem_dmrg_ccpvdz_path = path + "results_nwchem_dmrg_ccpvdz.dat"
pno_dmrg_path = path + "results_pno_dmrg.dat"


fci_fb_data = np.genfromtxt(fci_fb_path, names=True)
fci_opt_data = np.genfromtxt(fci_opt_path, names=True)
nwchem_dmrg_oo_data = np.genfromtxt(nwchem_dmrg_oo_path, names=True)    # oo for orbital optimization
pno_dmrg_oo_data = np.genfromtxt(pno_dmrg_oo_path, names=True)          # oo for orbital optimization
pno_fci_oo_data = np.genfromtxt(pno_fci_oo_path, names=True)            # oo for orbital optimization

nwchem_dmrg_631g_data = np.genfromtxt(nwchem_dmrg_631g_path, names=True)
nwchem_dmrg_ccpvdz_data = np.genfromtxt(nwchem_dmrg_ccpvdz_path, names=True)

pno_dmrg_data = np.genfromtxt(pno_dmrg_path, names=True)

fci_fb_distance = fci_fb_data['distance']
fci_fb_energy = fci_fb_data['FCI']

# With Orbital Optimization

mask_fci_opt = fci_opt_data['iteration'] == 5
fci_opt_distance = fci_opt_data['distance'][mask_fci_opt]
fci_opt_energy_0 = fci_opt_data['energy_0'][mask_fci_opt]

nwchem_dmrg_oo_distance = nwchem_dmrg_oo_data['distance']
nwchem_dmrg_oo_energy_0 = nwchem_dmrg_oo_data['energy_0']
nwchem_dmrg_oo_energy_1 = nwchem_dmrg_oo_data['energy_1']
nwchem_dmrg_oo_energy_2 = nwchem_dmrg_oo_data['energy_2']

pno_dmrg_oo_distance = pno_dmrg_oo_data['distance']
pno_dmrg_oo_energy_0 = pno_dmrg_oo_data['energy_0']
pno_dmrg_oo_energy_1 = pno_dmrg_oo_data['energy_1']
pno_dmrg_oo_energy_2 = pno_dmrg_oo_data['energy_2']

pno_fci_oo_distance = pno_fci_oo_data['distance']
pno_fci_oo_energy_0 = pno_fci_oo_data['energy_0']


# Without Orbital Optimization (for comparison)
nwchem_dmrg_631g_distance = nwchem_dmrg_631g_data['distance']
nwchem_dmrg_631g_energy_0 = nwchem_dmrg_631g_data['energy_0']
nwchem_dmrg_631g_energy_1 = nwchem_dmrg_631g_data['energy_1']
nwchem_dmrg_631g_energy_2 = nwchem_dmrg_631g_data['energy_2']

nwchem_dmrg_ccpvdz_distance = nwchem_dmrg_ccpvdz_data['distance']
nwchem_dmrg_ccpvdz_energy_0 = nwchem_dmrg_ccpvdz_data['energy_0']
nwchem_dmrg_ccpvdz_energy_1 = nwchem_dmrg_ccpvdz_data['energy_1']
nwchem_dmrg_ccpvdz_energy_2 = nwchem_dmrg_ccpvdz_data['energy_2']


pno_dmrg_distance = pno_dmrg_data['distance']
pno_dmrg_energy_0 = pno_dmrg_data['energy_0']
pno_dmrg_energy_1 = pno_dmrg_data['energy_1']
pno_dmrg_energy_2 = pno_dmrg_data['energy_2']

# Plot with Orbital Optimization
plt.figure(figsize=(8,5), dpi=300)
#plt.plot(fci_fb_distance, fci_fb_energy, color='tab:blue', label="6-31g FCI-fixed $E_0$")
#plt.plot(fci_opt_distance, fci_opt_energy_0, color='tab:green', label="6-31g FCI OO $E_0$")
plt.plot(nwchem_dmrg_oo_distance, nwchem_dmrg_oo_energy_0, color='tab:orange', label="6-31g DMRG OO $E_0$")
#plt.plot(pno_dmrg_oo_distance, pno_dmrg_oo_energy_0, color='tab:red', label="PNO DMRG OO $E_0$")
#plt.plot(pno_fci_oo_distance, pno_fci_oo_energy_0, color='tab:brown', label="PNO FCI OO $E_0$")


# plot excited states
plt.plot(nwchem_dmrg_oo_distance, nwchem_dmrg_oo_energy_1, color='tab:purple', label="6-31g DMRG OO $E_1$")
plt.plot(nwchem_dmrg_oo_distance, nwchem_dmrg_oo_energy_2, color='tab:pink', label="6-31g DMRG OO $E_2$")

#plt.plot(pno_dmrg_oo_distance, pno_dmrg_oo_energy_1, color='tab:olive', label="PNO DMRG OO $E_1$")
#plt.plot(pno_dmrg_oo_distance, pno_dmrg_oo_energy_2, color='tab:cyan', label="PNO DMRG OO $E_2$")

# Plot without Orbital Optimization (for comparison)
plt.plot(nwchem_dmrg_631g_distance, nwchem_dmrg_631g_energy_0, color='tab:orange', linestyle='dashed', label="6-31g DMRG $E_0$")
plt.plot(nwchem_dmrg_631g_distance, nwchem_dmrg_631g_energy_1, color='tab:purple', linestyle='dashed', label="6-31g DMRG $E_1$")
plt.plot(nwchem_dmrg_631g_distance, nwchem_dmrg_631g_energy_2, color='tab:pink', linestyle='dashed', label="6-31g DMRG $E_2$")

plt.plot(nwchem_dmrg_ccpvdz_distance, nwchem_dmrg_ccpvdz_energy_0, color='tab:orange', linestyle='dotted', label="cc-pVDZ DMRG $E_0$")
plt.plot(nwchem_dmrg_ccpvdz_distance, nwchem_dmrg_ccpvdz_energy_1, color='tab:purple', linestyle='dotted', label="cc-pVDZ DMRG $E_1$")
plt.plot(nwchem_dmrg_ccpvdz_distance, nwchem_dmrg_ccpvdz_energy_2, color='tab:pink', linestyle='dotted', label="cc-pVDZ DMRG $E_2$")
'''
plt.plot(pno_dmrg_distance, pno_dmrg_energy_0, color='tab:red', linestyle='dashed', label="PNO DMRG $E_0$")
plt.plot(pno_dmrg_distance, pno_dmrg_energy_1, color='tab:olive', linestyle='dashed', label="PNO DMRG $E_1$")
plt.plot(pno_dmrg_distance, pno_dmrg_energy_2, color='tab:cyan', linestyle='dashed', label="PNO DMRG $E_2$")

'''
#plt.xlabel("intermolecular distance d (Å)") # distance between two H2 molecules
plt.xlabel("internuclear distance d (Å)") # distance between equidistant H atoms
plt.ylabel("potential energy (eV)")
#plt.title("H₄ Linear Configuration: H–H (2.55 Å) … d … H–H (2.55 Å) - GS + ES")
plt.title ("H₄ Linear Configuration: Equidistant Spacing d - GS + ES")
plt.grid(True)
plt.legend(loc="best")
#plt.xlim(0.3, 2.5)
#plt.ylim(-1.2, -0.9)  # optional
plt.tight_layout()
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.show()
