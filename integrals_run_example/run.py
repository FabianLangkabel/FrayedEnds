import subprocess as sp
import sys
sys.path.append('/workspaces/MRA-OrbitalOptimization/build/integrals_nanobind')
import MRA_integrals

Integrator = MRA_integrals.Integrator(50.0, 7, 0.0001)
Integrator.ReadOrbitals()
Integrator.CalculateIntegrals()
# Referenz
programm = sp.call("/workspaces/MRA-OrbitalOptimization/build/integrals_ref/IntegralsRef", shell = True)