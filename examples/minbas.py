from madpy.minbas import AtomicBasisProjector
from madpy import Plotter

geom = "Li 0.0 0.0 -10\nH 0.0 0.0 10"
bp = AtomicBasisProjector(geom)

orbitals = bp.orbitals
del bp

plt=Plotter()
data = []
for i in range(len(orbitals)):
    plt.line_plot(f"atomic{i}.dat",orbitals[i])
del plt
