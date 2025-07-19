import numpy
from ._madpy_impl import PNOInterface
from .madworld import redirect_output, get_function_info
import glob
import os


class MadPNO:

    _orbitals = None
    _h = None # one-body tensor
    _g = None # two-body tensor
    _c = 0.0 # constant term
    impl = None
    _world = None

    @property
    def orbitals(self, *args, **kwargs):
        """
        Convenience
        """
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld, geometry, n_orbitals=None, no_compute=False, maxrank=None, diagonal=True, frozen_core=True, *args, **kwargs):
        # todo: replace geometry with instalce of molecule class (expose to python)
        if not no_compute and n_orbitals is None:
            raise Exception("madpno: n_orbitals needs to be set")

        # check if geometry is given as a file
        # if not write the file
        if not os.path.exists(geometry):
            self.create_molecule_file(geometry_angstrom=geometry)
            geometry="molecule"

        if maxrank is None:
            # safe option, with this we always compute enough pnos
            maxrank = n_orbitals
            # more effective
            try:
                from tequila.quantumchemistry import ParametersQC
                ne=ParametersQC(geometry=geometry).n_electrons
                np = ne//2
                maxrank = int(numpy.ceil(n_orbitals/np))

            except Exception:
                maxrank = n_orbitals

        self._world = madworld
        self._world.add_instance(self)
        pno_input_string = self.parameter_string(molecule_file=geometry, maxrank=maxrank, diagonal=diagonal, frozen_core=frozen_core,  *args, **kwargs)
        print(pno_input_string)

        self.impl = PNOInterface(self._world._impl, pno_input_string)

        if not no_compute:
            self._orbitals = self.compute_orbitals(n_orbitals=n_orbitals, *args, **kwargs)

    def get_pno_groupings(self, diagonal=True, *args, **kwargs):
        # group the PNOs according to their pair IDs. For diagonal approximation (default) this corresponds to SPA edges
        orbitals = self.get_orbitals(*args, **kwargs)
        info = get_function_info(orbitals)
        nhf = len([x for x in info if numpy.isclose(float(x["occ"]),2.0)])
        diagonal = {k:[] for k in range(nhf)}
        off_diagonal = {(k,l):[] for k in range(nhf) for l in range(k,nhf)}
        for k in range(len(orbitals)):
            x = info[k]["pair1"]
            y = info[k]["pair2"]
            if x == y: diagonal[x].append(k)
            else: off_diagonal[(x,y)].append(k)

        if diagonal:
            return diagonal
        return {**diagonal, **off_diagonal}

    def get_spa_edges(self, frozen_core=True):
        pno_groupings = self.get_pno_groupings(diagonal=True)
        edges = [tuple(sorted(x)) for x in pno_groupings.values()]
        if frozen_core:
            orbitals = self.get_orbitals()
            info = get_function_info(orbitals)
            # indices of hf orbitals that are frozen and
            occf = [k for k,x in enumerate(info) if numpy.isclose(float(x["occ"]), 2.0) and "frozen" in x["type"]]
            # compute offset
            nof = len(occf)
            if nof==0: return edges

            if not all([k == i for i,k in enumerate(occf)]):
                raise Exception("get_spa_edges with frozen_core=True only works for occupied frozen orbitals consecutively numbered starting with 0, here we have: {}".format(str(occf)))

            # remove frozen orbitals
            edges = [edge for edge in edges if len(edge)!=0 and edge[0] not in occf]
            # correct edges with offset
            edges = [tuple([y-nof for y in x]) for x in edges]
        return edges

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            raise Exception("orbitals not yet computed")

    def get_nuclear_potential(self, *args, **kwargs):
        return self.impl.get_nuclear_potential()

    def get_nuclear_repulsion(self, *args, **kwargs):
        return self.impl.get_nuclear_repulsion()

    def get_sto3g(self, *args, **kwargs):
        return self.impl.get_sto3g()

    @redirect_output("madpno.log")
    def compute_orbitals(self, n_orbitals, frozen_virt_dim=0, *args, **kwargs):
        self.impl.run(n_orbitals)
        frozen_occ_dim = self.impl.get_frozen_core_dim()
        active_dim = n_orbitals - frozen_occ_dim - frozen_virt_dim
        # package the orbitals
        orbitals = self.impl.GetPNOs(frozen_occ_dim, active_dim, frozen_virt_dim)
        self.cleanup(*args, **kwargs)
        self._orbitals = orbitals
        return orbitals

    def compute_integrals(self, *args, **kwargs):
        if self._orbitals is None:
            self.compute_orbitals(*args, **kwargs)
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

        data["dft"] = {"xc": "hf", "L":self._world.L,  "k": self._world.k, "econv": 1.e-4, "dconv": 5.e-4, "localize": "boys", "ncf": "( none , 1.0 )"}
        data["pno"] = {"maxrank": maxrank, "f12": "false", "thresh": 1.e-4, "diagonal": diagonal}
        # this should be gone soon
        data["pnoint"] = {"n_pno": 10, "orthog": "symmetric"}

        if not frozen_core:
            data["pno"]["freeze"] = 0
        data["plot"] = {}
        for key in data.keys():
            if key in kwargs:
                data[key] = {**data[key], **kwargs[key]}

        input_str = "pno --geometry=\"source_type=inputfile; units=angstrom; no_orient=1; eprec=1.e-6; source_name=" + molecule_file + "\""
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

    def create_molecule_file(self, geometry_angstrom, filename="molecule"):
        molecule_file_str = "geometry\n"
        molecule_file_str += geometry_angstrom
        molecule_file_str += "\nend"
        molecule_file_str = os.linesep.join([s for s in molecule_file_str.splitlines() if s])
        f = open(filename, "w")
        f.write(molecule_file_str)
        f.close()

    def cleanup(*args, **kwargs):
        # Define the patterns for the files to delete
        patterns = [
            "*.00000",                # Files ending with .00000
            "N7madness*",             # Files starting with N7madness
            "mad.calc_info.json",     # Specific file
            "mad.restartaodata",      # Specific file
            "pnoinfo.txt"             # Specific file
        ]

        # Iterate over each pattern and delete matching files
        for pattern in patterns:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except OSError as e:
                    print(f"Error deleting {file}: {e}")

