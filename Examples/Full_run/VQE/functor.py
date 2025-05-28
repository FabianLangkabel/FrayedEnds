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
import sys
sys.path.append('/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/madness_extension')
import MadPy as mad

def test(x:float) -> float:
    return x**2

F=mad.return_f()
print(F(2.0))  
print(type(F))

print(mad.call_python_function(test,2.0))

F=mad.Functor(test)
print(mad.test_function(F,2.0))
del F

MRafunc=mad.CreateFunc(50,7,0.0001,test)
