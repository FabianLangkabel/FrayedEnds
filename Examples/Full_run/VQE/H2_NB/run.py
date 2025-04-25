import tequila as tq
import numpy as np
import os
import shutil
from pathlib import Path
import logging
import subprocess as sp
import pyscf
from pyscf import fci
import OrbOpt_helper
import sys
sys.path.append('/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/madness_extension')
import MadPy as mad

distance = 2.5 # Distance between the two hydrogen atoms in Bohr 
iteration_energies = [] #Stores the energies at the beginning of each iteration step after the VQE
all_occ_number = [] #Stores the orbital occupations at the beginning of each iteration step after the VQE
iterations = 6
molecule_name = "h2"
box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.0001
optimization_thresh = 0.001
NO_occupation_thresh = 0.001

test=mad.PNOInterface(2,"a help=isontheeway help",box_size,wavelet_order,madness_thresh)
test.run()

