import os
from .madworld import redirect_output
from ._madpy_impl import MinBasProjector


class AtomicBasisProjector:

    impl = None

    def __init__(self, madworld, geometry, units=None, aobasis="sto-3g", *args, **kwargs):
        # check if geometry is given as a file
        # if not write the file
        if not os.path.exists(geometry):
            self.create_molecule_file(geometry_angstrom=geometry)
            geometry = "molecule"
        
        if units is None:
                if self.silent==False:
                    print("Warning: No units passed with geometry, assuming units are angstrom.")
                units = "angstrom"
        else:
            units = units.lower()
            if units in ["angstrom", "ang", "a", "å"]:
                units = "angstrom"
            elif units in ["bohr", "atomic units", "au", "a.u."]:
                units = "bohr"
            else:
                if self.silent==False:
                    print("Warning: Units passed with geometry not recognized (available units are angstrom or bohr), assuming units are angstrom.")
                units = "angstrom"
        
        input_string = self.parameter_string(
            madworld, molecule_file=geometry, units=units, aobasis=aobasis, *args, **kwargs
        )
        print(input_string)

        self.impl = MinBasProjector(madworld._impl, input_string)

        self.impl.run()
        orbitals = self.impl.get_atomic_basis()
        self.orbitals = orbitals

    def get_nuclear_repulsion(self):
        return self.impl.get_nuclear_repulsion()

    @redirect_output("minbas_scf.log")
    def solve_scf(self, thresh=1.e-4):
        return self.impl.solve_scf(thresh)

    def get_nuclear_potential(self):
        return self.impl.get_nuclear_potential()

    def parameter_string(
        self, madworld, molecule_file, units, aobasis="sto-3g", **kwargs
    ) -> str:
        data = {}

        data["dft"] = {
            "xc": "hf",
            "L": madworld.L,
            "k": madworld.k,
            "econv": 1.0e-4,
            "dconv": 5.0e-4,
            "localize": "boys",
            "ncf": "( none , 1.0 )",
            "aobasis": "sto-3g",
        }

        if units=="bohr":
            input_str = (
                'dft --geometry="source_type=inputfile; units=bohr; no_orient=1; eprec=1.e-6; source_name='
                + molecule_file
                + '"'
            )
        else:
            input_str = (
                'dft --geometry="source_type=inputfile; units=angstrom; no_orient=1; eprec=1.e-6; source_name='
                + molecule_file
                + '"'
            )
        input_str += ' --dft="'
        for k, v in data["dft"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + '"'

        return input_str

    def create_molecule_file(self, geometry_angstrom, filename="molecule"):
        molecule_file_str = "molecule\n"
        molecule_file_str += geometry_angstrom
        molecule_file_str += "\nend"
        molecule_file_str = os.linesep.join(
            [s for s in molecule_file_str.splitlines() if s]
        )
        f = open(filename, "w")
        f.write(molecule_file_str)
        f.close()
