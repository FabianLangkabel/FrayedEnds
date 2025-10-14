import madpy as mad
from madpy.minbas import AtomicBasisProjector

geom = "Li 0.0 0.0 -10\nH 0.0 0.0 10"

world = mad.MadWorld3D()

bp = AtomicBasisProjector(world, geom)

orbitals = bp.orbitals

data = []
for i in range(len(orbitals)):
    world.line_plot(f"atomic{i}.dat", orbitals[i])

mad.cleanup(globals())
