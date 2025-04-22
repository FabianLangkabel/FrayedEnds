import subprocess as sp
import sys
sys.path.append('/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/integrals_nanobind')
import MRA_integrals

for i in range(3):
    print("nanobind run: ", i)
    Integrator = MRA_integrals.Integrator(50.0, 7, 0.0001)
    Integrator.ReadOrbitals()
    Integrator.CalculateIntegrals()
    del Integrator
# Referenz
print("Reference run:")
programm = sp.call("/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/integrals_ref/IntegralsRef", shell = True)