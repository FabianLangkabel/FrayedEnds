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

def test(x:float, y:float, z:float) -> float:
    return x**2 + y**2 + z**2

factory=mad.PyFuncFactory(50,7,0.0001,test)
mraf=factory.GetMRAFunction()
del factory
print("MRA function created successfully.")