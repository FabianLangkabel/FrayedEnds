import frayedends as fe
import os
import numpy as np

input_dir = "/Users/timo/workspace/exc_dir/job_10223339/hexacene_pi"

def load_orbitals(path, name):
    result = []
    didntbreakyet = True
    i = 0
    while didntbreakyet:
        filename = os.path.join(path, f"{name}{i}.fe")
        try:
            result.append(fe.SavedFct3D(filename))
            i += 1
        except:
            didntbreakyet = False
    return result


world = fe.MadWorld(ndims=3, thresh=1e-6)

nwchem_prefix = os.path.join(input_dir, "nwchem/hexacene")

converter = fe.NWChem_Converter(world)
converter.read_nwchem_file(nwchem_prefix)
nwchem_mos = converter.get_mos()
nwchem_occupancies = np.array(converter.get_occupancies())
nwchem_energies = np.array(converter.get_orbital_energies())
# the molecule is taken from the NWChem output to guarantee the same coordinate frame for all orbitals
molecule = converter.get_molecular_geometry()


hf_orbitals = load_orbitals(os.path.join(input_dir, "hf_nw_orbs"), "core_")

for i in range(len(hf_orbitals)):
    world.cube_plot(f"hf_orb{i}", hf_orbitals[i], molecule, zoom=3.0)

for i in range(len(nwchem_mos)):
    world.cube_plot(f"nwchem_orb{i}", nwchem_mos[i], molecule, zoom=3.0)

fe.cleanup(globals())