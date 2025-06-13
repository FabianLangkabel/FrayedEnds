import tequila as tq
import numpy as np
import os
import shutil
from pathlib import Path
import logging
import subprocess as sp
import pyscf
from pyscf import fci
import time
import madpy as mad

class Coulomb_with_param:
    epsilon = 0.0,
    def __init__(self,epsilon):
        self.epsilon=epsilon
    def test(self, x, y, z):
        v=np.array([x, y, z])
        return -1/np.sqrt(v@v+self.epsilon**2)

tt=Coulomb_with_param(0.0)
print(tt.test(1,2,3))
factory=mad.PyFuncFactory(50,7,0.0001,tt.test)
mraf=factory.GetMRAFunction()
del factory
print("MRA function created successfully.")

liste=["yz","xy","xz"]
opti=mad.Optimization(50,7,0.0001)
for i in range(3):
    opti.plot("custom_func"+str(i)+".dat",mraf,axis=i)
opti.plane_plot(".dat",mraf,plane="xy",zoom=5.0,datapoints=81,origin=[0.0,0.0,0.0])
del opti