from ._madpy_impl import PNOInterface
from .parameters import redirect_output
from .baseclass import MadPyBase

import os

class MadPNO(MadPyBase):

    _orbitals = None
    _h = None # one-body tensor
    _g = None # two-body tensor
    _c = 0.0 # constant term

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if geometry is given as a file
        # if not write the file
        if not os.path.exists(geometry):
            self.create_molecule_file(geometry_bohr=geometry)
            geometry="molecule"

        pno_input_string = self.parameter_string(molecule_file=geometry, *args, **kwargs)
        print(pno_input_string)
        self.impl = PNOInterface(pno_input_string, self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh)


    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            self.compute_pnos()
            return self.get_orbitals()

    def get_integrals(self, *args, **kwargs):
        if self._h is not None and self._g is not None:
            return self._c, self._h, self._g
        else:
            self.compute_integrals(*args, **kwargs)
            return self.get_integrals(*args, **kwargs)

    def get_nuclear_potential(self, *args, **kwargs):
        return self.impl.get_nuclear_potential()

    def get_nuclear_repulsion(self, *args, **kwargs):
        return self.impl.GetNuclearRepulsion()

    def __call__(self, *args, **kwargs):
        self._orbitals = self.compute_pnos()
        c,h,g = self.compute_integrals(*args, **kwargs)
        return self._c,self._h,self._g

    @redirect_output("madpno.log")
    def compute_pnos(self, *args, **kwargs):
        self.impl.DeterminePNOsAndIntegrals()
        self._orbitals = self.impl.GetPNOs(0,2,0)  # input: dimensions of (frozen_occ, active, forzen_virt) space
        return self._orbitals

    @redirect_output("madpno2.log")
    def compute_integrals(self, *args, **kwargs):
        if self._orbitals is None:
            self.compute_pnos()
        self._h = self.impl.GetHTensor()
        self._g = self.impl.GetGTensor()
        self._c = self.impl.GetNuclearRepulsion()
        return self._c,self._h,self._g

    def parameter_string(self, molecule_file, maxrank=10, diagonal=True, frozen_core=True, **kwargs) -> str:
        """
        :param molecule_file: file containing the molecular coordinates
        :param maxrank: maxrank for each set of PNOs
        :param diagonal: use diagonal approximation (default True)
        :param frozen_core: use frozen core approximation (default True)
        :param kwargs: additional key/value pairs. Example kwargs={"dft":{"k":5, "econv":1.e-6}, "pno":{...}, ... }
        :return: parameter string for the PNO class in madness
        """

        data = {}

        data["dft"] = {"xc": "hf", "L":self.madness_parameters.L,  "k": self.madness_parameters.k, "econv": 1.e-4, "dconv": 5.e-4, "localize": "boys", "ncf": "( none , 1.0 )"}
        data["pno"] = {"maxrank": maxrank, "f12": "false", "thresh": 1.e-4, "diagonal": diagonal}
        # this should be gone soon
        data["pnoint"] = {"n_pno": 10, "orthog": "symmetric"}

        if not frozen_core:
            data["pno"]["freeze"] = 0
        data["plot"] = {}
        for key in data.keys():
            if key in kwargs:
                data[key] = {**data[key], **kwargs[key]}

        input_str = "pno --geometry=\"source_type=inputfile; no_orient=1; source_name=" + molecule_file + "\""
        input_str += " --dft=\""
        for k, v in data["dft"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + "\""
        input_str += " --pno=\""
        for k, v in data["pno"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + "\""
        input_str += " --pnoint=\""
        for k, v in data["pnoint"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + "\""
        if data["plot"] != {}:
            input_str += " --plot=\""
            for k, v in data["plot"].items():
                input_str += "{}={}; ".format(k, v)
            input_str = input_str[:-2] + "\""

        return input_str

    def create_molecule_file(self, geometry_bohr, filename="molecule"):
        molecule_file_str = """geometry
    units Bohr
    no_orient 1
    eprec 1.e-6\n"""
        molecule_file_str += geometry_bohr
        molecule_file_str += "\nend"
        molecule_file_str = os.linesep.join([s for s in molecule_file_str.splitlines() if s])
        f = open(filename, "w")
        f.write(molecule_file_str)
        f.close()

