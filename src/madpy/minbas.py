import numpy
from ._madpy_impl import MinBasProjector
from  .madpno import MadPNO
from .parameters import redirect_output
from .baseclass import MadPyBase, get_function_info
import glob
import os

class AtomicBasisProjector(MadPyBase):

    def __init__(self, geometry, aobasis="sto-3g",  *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check if geometry is given as a file
        # if not write the file
        if not os.path.exists(geometry):
            self.create_molecule_file(geometry_angstrom=geometry)
            geometry="molecule"

        input_string = self.parameter_string(molecule_file=geometry, aobasis=aobasis, *args, **kwargs)
        print(input_string)
        self.impl = MinBasProjector(input_string, self.madness_parameters.L, self.madness_parameters.k,
                                 self.madness_parameters.thresh, self.madness_parameters.initial_level,
                                 self.madness_parameters.truncate_mode, self.madness_parameters.refine,
                                 self.madness_parameters.n_threads)
        print("calling run")
        self.impl.run()
        print("done")
        orbitals = self.impl.get_atomic_basis()
        self.orbitals = orbitals

    def get_nuclear_repulsion(self):
        return self.impl.get_nuclear_repulsion()
    def get_nuclear_potential(self):
        return self.impl.get_nuclear_potential()

    def parameter_string(self, molecule_file, aobasis="sto-3g", **kwargs) -> str:
            data = {}

            data["dft"] = {"xc": "hf", "L": self.madness_parameters.L, "k": self.madness_parameters.k, "econv": 1.e-4,
                           "dconv": 5.e-4, "localize": "boys", "ncf": "( none , 1.0 )", "aobasis": "sto-3g"}

            input_str = "dft --geometry=\"source_type=inputfile; units=angstrom; no_orient=1; eprec=1.e-6; source_name=" + molecule_file + "\""
            input_str += " --dft=\""
            for k, v in data["dft"].items():
                input_str += "{}={}; ".format(k, v)
            input_str = input_str[:-2] + "\""

            return input_str

    def create_molecule_file(self, geometry_angstrom, filename="molecule"):
            molecule_file_str = "geometry\n"
            molecule_file_str += geometry_angstrom
            molecule_file_str += "\nend"
            molecule_file_str = os.linesep.join([s for s in molecule_file_str.splitlines() if s])
            f = open(filename, "w")
            f.write(molecule_file_str)
            f.close()



