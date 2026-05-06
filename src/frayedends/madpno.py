import glob
import os

import numpy

from ._frayedends_impl import PNOInterface
from .madworld import get_function_info, redirect_output
from .moleculargeometry import MolecularGeometry


class MadPNO:
    _gs_orbitals = None # ground state orbitals (HF + MP2 PNOs)
    _ex_orbitals = None # excited state orbitals (CIS X vectors + CISPD PNOs)
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    impl = None

    @property
    def gs_orbitals(self, *args, **kwargs):
        """
        Convenience access for ground state orbitals
        """
        return self.get_gs_orbitals(*args, **kwargs)

    @property
    def ex_orbitals(self, *args, **kwargs):
        """
        Convenience access for excited state orbitals
        """
        return self.get_ex_orbitals(*args, **kwargs)

    def __init__(
        self,
        madworld,
        geometry,
        units=None,
        silent=False,
        n_orbitals=None,
        no_compute=False,
        maxrank=None,
        diagonal=True,
        frozen_core=True,
        cispd=-1,
        *args,
        **kwargs,
    ):
        self.silent = silent
        # todo: replace geometry with instance of molecule class (expose to python)
        if not no_compute and n_orbitals is None:
            raise Exception("madpno: n_orbitals needs to be set")

        if maxrank is None:
            molgeom = MolecularGeometry(geometry, units=units)
            n_tot_e = molgeom.n_electrons
            if frozen_core:
                n_act_e = n_tot_e - molgeom.n_core_electrons
            else:
                n_act_e = n_tot_e
            n_hf_orbs = n_tot_e / 2
            n_act_pairs = n_act_e / 2
            maxrank = int(numpy.ceil((n_orbitals - n_hf_orbs) / n_act_pairs))

        # check if geometry is given as a file
        # if not write the file
        if not os.path.exists(geometry):
            self.create_molecule_file(geometry=geometry)
            geometry = "molecule"

        if units is None:
            if not self.silent:
                print("Warning: No units passed with geometry, assuming units are angstrom.")
            units = "angstrom"
        else:
            units = units.lower()
            if units in ["angstrom", "ang", "a", "å"]:
                units = "angstrom"
            elif units in ["bohr", "atomic", "atomic units", "au", "a.u."]:
                units = "bohr"
            else:
                if not self.silent:
                    print(
                        "Warning: Units passed with geometry not recognized (available units are angstrom or bohr), assuming units are angstrom."
                    )
                units = "angstrom"

        pno_input_string = self.parameter_string(
            madworld,
            molecule_file=geometry,
            units=units,
            maxrank=maxrank,
            diagonal=diagonal,
            frozen_core=frozen_core,
            cispd=cispd,
            *args,
            **kwargs,
        )
        print(f"DEBUG: MADNESS-String: {pno_input_string}")
        self.impl = PNOInterface(madworld.impl, pno_input_string)

        if not no_compute:
            self.compute_orbitals(n_orbitals=n_orbitals, *args, **kwargs)

    @redirect_output("madpno.log")
    def compute_orbitals(self, n_orbitals, *args, **kwargs):
        self.impl.run(n_orbitals)
        # package the orbitals
        self._gs_orbitals = self.impl.get_gs_orbitals()
        self._ex_orbitals = self.impl.get_ex_orbitals()
        self.cleanup(*args, **kwargs)

    def get_gs_orbitals(self, *args, **kwargs):
        if self._gs_orbitals is not None:
            return self._gs_orbitals
        else:
            raise Exception("ground state orbitals not yet computed")

    def get_ex_orbitals(self, *args, **kwargs):
        if self._ex_orbitals is not None:
            return self._ex_orbitals
        else:
            raise Exception("excited state orbitals not yet computed")

    def get_pno_groupings(self, diagonal=True, *args, **kwargs):
        # group the PNOs according to their pair IDs. For diagonal approximation (default) this corresponds to SPA edges
        orbitals = self.get_gs_orbitals(*args, **kwargs)
        info = get_function_info(orbitals)
        nhf = len([x for x in info if numpy.isclose(float(x["occ"]), 2.0)])
        diagonal = {k: [] for k in range(nhf)}
        off_diagonal = {(k, l): [] for k in range(nhf) for l in range(k, nhf)}
        for k in range(len(orbitals)):
            x = info[k]["pair1"]
            y = info[k]["pair2"]
            if x == y:
                diagonal[x].append(k)
            else:
                off_diagonal[(x, y)].append(k)

        if diagonal:
            return diagonal
        return {**diagonal, **off_diagonal}

    def get_spa_edges(self, frozen_core=True):
        pno_groupings = self.get_pno_groupings(diagonal=True)
        edges = [tuple(sorted(x)) for x in pno_groupings.values()]
        nfreeze = self.impl.get_frozen_core_dim()
        if frozen_core:
            orbitals = self.get_gs_orbitals()
            info = get_function_info(orbitals)
            # indices of hf orbitals that are frozen and
            occf = [k for k, x in enumerate(info) if numpy.isclose(float(x["occ"]), 2.0) and k < nfreeze]
            # compute offset
            nof = len(occf)
            if nof == 0:
                return edges

            if not all([k == i for i, k in enumerate(occf)]):
                raise Exception(
                    "get_spa_edges with frozen_core=True only works for occupied frozen orbitals consecutively numbered starting with 0, here we have: {}".format(
                        str(occf)
                    )
                )

            # remove frozen orbitals
            edges = [edge for edge in edges if len(edge) != 0 and edge[0] not in occf]
            # correct edges with offset
            edges = [tuple([y - nof for y in x]) for x in edges]
        return edges

    def get_nuclear_potential(self, *args, **kwargs):
        return self.impl.get_nuclear_potential()

    def get_nuclear_repulsion(self, *args, **kwargs):
        return self.impl.get_nuclear_repulsion()

    def get_sto3g(self, *args, **kwargs):
        return self.impl.get_sto3g()

    def parameter_string(
        self,
        madworld,
        molecule_file,
        units,
        maxrank=10,
        diagonal=True,
        frozen_core=True,
        cispd=-1,
        **kwargs,
    ) -> str:
        """
        :param molecule_file: file containing the molecular coordinates
        :param maxrank: maxrank for each set of PNOs
        :param diagonal: use diagonal approximation (default True)
        :param frozen_core: use frozen core approximation (default True)
        :param cispd: number of excited states for CIS(D) PNO generation (default -1)
        :param kwargs: additional key/value pairs. Example kwargs={"dft":{"k":5, "econv":1.e-6}, "pno":{...}, ... }
        :return: parameter string for the PNO class in madness
        """

        data = {}

        data["dft"] = {
            "xc": "hf",
            "L": madworld.get_function_defaults()["cell_width"] / 2,
            "k": madworld.get_function_defaults()["k"],
            "econv": 1.0e-6,
            "dconv": 5.0e-4,
            "localize": "boys",
        }
        data["nemo"] = {"ncf": "( none , 1.0)"}
        if cispd > -1:
            data["tdhf"] = {
                "nexcitations": cispd + 1,
                "thresh": 1.0e-5,
                "restart": "no_restart"
            }

        data["pno"] = {
            "maxrank": maxrank,
            "f12": "false",
            "thresh": madworld.get_function_defaults()["thresh"],
            "diagonal": diagonal,
        }

        if not frozen_core:
            data["pno"]["freeze"] = 0
        data["plot"] = {}
        for key in data.keys():
            if key in kwargs:
                data[key] = {**data[key], **kwargs[key]}
        if units == "bohr":
            input_str = (
                'pno --geometry="source_type=inputfile; units=bohr; no_orient=1; eprec=1.e-6; source_name='
                + molecule_file
                + '"'
            )
        else:
            input_str = (
                'pno --geometry="source_type=inputfile; units=angstrom; no_orient=1; eprec=1.e-6; source_name='
                + molecule_file
                + '"'
            )

        for item in ["dft", "pno", "nemo", "tdhf", "plot"]:
            if item in data and data[item]:
                input_str += ' --{}="'.format(item)
                for k, v in data[item].items():
                    input_str += "{}={}; ".format(k, v)
                input_str = input_str[:-2] + '"'

        return input_str

    def create_molecule_file(self, geometry, filename="molecule"):
        molecule_file_str = "molecule\n"
        molecule_file_str += geometry
        molecule_file_str += "\nend"
        molecule_file_str = os.linesep.join([s for s in molecule_file_str.splitlines() if s])
        f = open(filename, "w")
        f.write(molecule_file_str)
        f.close()

    def cleanup(*args, **kwargs):
        # Define the patterns for the files to delete
        patterns = [
            "*.00000",  # Files ending with .00000
            "MacroTask*",  # Files starting with MacroTask
            "N7madness7*",  # Files starting with N7madness7
            "mad.calc_info.json",  # Specific file
            "mad.restartaodata",  # Specific file
            "pnoinfo.txt",  # Specific file
        ]

        # Iterate over each pattern and delete matching files
        for pattern in patterns:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {file}: {e}")
