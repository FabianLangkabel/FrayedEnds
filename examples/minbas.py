import frayedends as fe
from frayedends.minbas import AtomicBasisProjector

geom = "Li 0.0 0.0 -10\nH 0.0 0.0 10"

world = fe.MadWorld3D()

bp = AtomicBasisProjector(world, geom)

orbitals = bp.orbitals

data = []
for i in range(len(orbitals)):
    world.line_plot(f"atomic{i}.dat", orbitals[i])

del bp
del world
