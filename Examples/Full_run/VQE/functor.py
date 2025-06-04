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


def test(x, y, z):
    v=np.array([x, y, z])
    q0=np.array([0.0, 0.0, -2.5])
    q1=np.array([0.0, 0.0, 2.5])
    return -np.exp(-0.1*(v-q0)@(v-q0)) - np.exp(-0.1*(v-q1)@(v-q1))


print(test(1,2,3))
factory=mad.PyFuncFactory(50,7,0.0001,test)
mraf=factory.GetMRAFunction()
del factory
print("MRA function created successfully.")

liste=["yz","xy","xz"]
opti=mad.Optimization(50,7,0.0001)
for i in range(3):
    opti.plot("custom_func"+str(i)+".dat",mraf,axis=i)
opti.plane_plot(".dat",mraf,plane="xy",zoom=5.0,datapoints=81,origin=[0.0,0.0,0.0])