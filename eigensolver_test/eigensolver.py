import sys
import time
import madpy as mad

start_time = time.time()

box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.0001
optimization_thresh = 0.001
NO_occupation_thresh = 0.001

molecule_name = "h2"
distance = 2.5 # Distance between the two hydrogen atoms in Bohr
distance = distance/2
geometry_bohr = '''
H 0.0 0.0 ''' + distance.__str__() + '''
H 0.0 0.0 ''' + (-distance).__str__()


peak_loc=[[0.0,0.0,-distance],[0.0,0.0,distance]] #locations of the peaks
sharpness_list=[100.0,100.0] #sharpness of the peaks
Q=2

mad_process = mad.MadnessProcess(box_size, wavelet_order, madness_thresh) # Initialize the Madness process

PotMaker = mad.CoulombPotentialFromChargeDensity(mad_process,sharpness_list,Q,peak_loc)
custom_pot=PotMaker.CreatePotential()
mad_process.plot("custom_potential.dat", custom_pot) #Plot the custom potential
del PotMaker

eigensolver = mad.Eigensolver(mad_process)
eigensolver.solve(custom_pot, 4, 100)
orbs=eigensolver.GetOrbitals(0,2,0)
del eigensolver
print(type(orbs))
print(type(orbs[0]))

del mad_process