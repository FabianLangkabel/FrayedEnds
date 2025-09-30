import madpy
import numpy as np
import tequila as tq
import time
import scipy as sp


geometry = "H 0.0 0.0 -1.1\nH 0.0 0.0 1.7"
molg=madpy.MolecularGeometry(geometry,units="angstrom")
print(molg.to_json())
print(4*sp.constants.pi*sp.constants.epsilon_0*sp.constants.hbar**2/(sp.constants.e**2*sp.constants.m_e)*1e10)
f=-1.1
print(f)