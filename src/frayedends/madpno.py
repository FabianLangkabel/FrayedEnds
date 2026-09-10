import glob
import os

import numpy

from ._frayedends_impl import PNOInterface
from .madworld import get_function_info, redirect_output
from .moleculargeometry import MolecularGeometry


class MadPNO:
    _orbitals = None # ground state orbitals (HF + MP2 PNOs) 
    _hf_orbitals = None # HF orbitals
    _cis_per_root = None # CIS X functions per root 
    _cis_orbitals = None # flat list with CIS X functions
    _cispd_orbitals = None # CISPD PNOs for excited states
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    impl = None

    @property
    def orbitals(self, *args, **kwargs):
        """
        Convenience access for ground state orbitals
        """
        return self.get_orbitals(*args, **kwargs)
    
    @property
    def hf_orbitals(self, *args, **kwargs):
        """
        Returns HF orbitals 
        """
        return self.get_hf_orbitals(*args, *kwargs)

    @property
    def cis_per_root(self):
        """
        CIS X vectors per root for excited states
        """
        if self._cis_per_root is not None:
            return self._cis_per_root
        raise Exception("CIS orbitals not yet computed. Call compute_cis() first.")
    
    @property
    def cis_orbitals(self):
        """
        flat CIS X vector for excited states
        """
        if self._cis_orbitals is not None:
            return self._cis_orbitals
        raise Exception("CIS orbitals not yet computed. Call compute_cis() first and then orthonormalize_cis().")
    
    @property
    def cispd_orbitals(self):
        """
        CISPD PNOs for excited states
        """
        if self._cispd_orbitals is not None:
            return self._cispd_orbitals
        raise Exception("CISPD orbitals not yet computed. Call compute_cispd() first.")

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
        *args,
        **kwargs,
    ):
        if madworld.dimensions != 3:
            raise ValueError(
                f"PNO calculation only possible in 3 dimensions. MadWorld is initialized with {madworld.dimensions} dims."
            )
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
            *args,
            **kwargs,
        )

        self.impl = PNOInterface(madworld.impl, pno_input_string)

        if not no_compute:
            self.compute_orbitals(n_orbitals=n_orbitals, *args, **kwargs)

    @redirect_output("madpno.log")
    def compute_orbitals(self, n_orbitals, *args, **kwargs):
        # Calculate HF + MP2 PNOs (ground state orbitals)
        self.impl.run(n_orbitals)
        # package the orbitals
        self._orbitals = self.impl.get_orbitals()
        self._hf_orbitals = self.impl.get_hf_orbitals()
        self.cleanup(*args, **kwargs)

    def _filter_best_contributions(self, cis_per_root, rtol=1e-2, atol=1e-8):
        # Filter CIS functions per excitation, keeping only those with largest contribution
        filtered = []
        for root in cis_per_root:
            if not root:
                filtered.append([])
                continue
            info = get_function_info(root)
            norms = [float(x.get("norm2", 0.0)) for x in info]
            max_norm2 = max(norms)
            best = [fct for fct, n in zip(root, norms) if numpy.isclose(n, max_norm2, rtol=rtol, atol=atol)]
            filtered.append(best)
        return filtered

    @redirect_output("cis.log")
    def compute_cis(self, n_excitation, dominant_contribution=False, *args, **kwargs):
        # Compute cis x functions 
        if self._orbitals is None:
            raise Exception("compute_orbitals() must be called before compute_cis()")
        self.impl.compute_cis(n_excitation)
        self._cis_per_root = self.impl.get_cis_x_per_root() # cis_per_root is a vector<vector<real_function_3d>>

        total_before = sum(len(root) for root in self._cis_per_root)

        if dominant_contribution:
            self._cis_per_root = self._filter_best_contributions(self._cis_per_root)
            total_after = sum(len(root) for root in self._cis_per_root)
            print(f"\n \n Dominant CIS functions kept: {total_after}/{total_before}")
            for ex, root in enumerate(self._cis_per_root):
                print(f"  Excitation {ex}: {len(root)} dominant function(s)\n ")

        cis_flat = [] 
        for root in self._cis_per_root:
            cis_flat.extend(root)   # created flat cis_orbitals (vector<real_function_3d>)
        
        self._cis_orbitals = cis_flat
        
        print("\nSelected CIS functions info:")
        for info in get_function_info(self._cis_orbitals):
            print(f"  type={info['type']} occ={info['occ']} pair1={info['pair1']} pair2={info['pair2']} norm2={info['norm2']:.6e}")
        print()

        return self._cis_orbitals
    
    @redirect_output("cispd.log")
    def compute_cispd(self, n_orbitals, dominant_contribution=False, *args, **kwargs):
        if self._cis_per_root is None:
            raise Exception("compute_cis() must be called before compute_cispd()")
        self.impl.compute_cispd(n_orbitals)
        raw_cispd = self.impl.get_cispd_orbitals()

        if dominant_contribution and self._cis_orbitals is not None:
            active_cis_info = get_function_info(self._cis_orbitals)
            active_pairs = []
            for x in active_cis_info:
                if x["type"].startswith("CIS_X_EX"):
                    ex_id = int(x["type"].split("_")[2][2:])
                    orb_id = int(x["pair1"])
                    pair = (ex_id, orb_id)
                    if pair not in active_pairs:
                        active_pairs.append(pair)

            cispd_info = get_function_info(raw_cispd)
            filtered_cispd = []
            for orb, x in zip(raw_cispd, cispd_info):
                if x["type"].startswith("CISPD_EX"):
                    ex_id = int(x["type"].split("_")[-1][2:])
                    p1, p2 = int(x["pair1"]), int(x["pair2"])
                    
                    if (ex_id, p1) in active_pairs or (ex_id, p2) in active_pairs:
                        filtered_cispd.append(orb)
            self._cispd_orbitals = filtered_cispd
            print(f"CISPD PNOs filtered: {len(filtered_cispd)}/{len(raw_cispd)} kept.")
        else:
            self._cispd_orbitals = raw_cispd

        print("\nSelected CISPD PNOs info:")
        for info in get_function_info(self._cispd_orbitals):
            print(f"  type={info['type']} occ={info['occ']} pair1={info['pair1']} pair2={info['pair2']}")
        print()

        self.cleanup(*args, **kwargs)
        return self._cispd_orbitals

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            raise Exception("ground state orbitals not yet computed")
    
    def get_hf_orbitals(self, *args, **kwargs):
        if self._hf_orbitals is not None:
            return self._hf_orbitals
        else: 
            raise Exception("ground state orbitals not yet computed")

    def _split_orbitals(self, orbitals):
        info = get_function_info(orbitals)
        gs_orbitals, ex_orbitals = [], []
        gs_indices, ex_indices = [], []
        for i, orb in enumerate(orbitals):
            label = info[i]["type"]
            if label.startswith("CIS_X_EX") or label.startswith("CISPD_EX"):
                ex_orbitals.append(orb)
                ex_indices.append(i)
            else:
                gs_orbitals.append(orb)
                gs_indices.append(i)
        return gs_orbitals, ex_orbitals, gs_indices, ex_indices

    def get_pno_groupings(self, diagonal=True, orbitals=None, *args, **kwargs):
        # group the PNOs according to their pair IDs. For diagonal approximation (default) this corresponds to SPA edges
        use_diagonal = diagonal

        if orbitals is None:
            orbitals = self.get_orbitals(*args, **kwargs)
            ex_orbitals = []
            if self._cis_orbitals is not None and self._cispd_orbitals is not None:
                ex_orbitals = self._cis_orbitals + self._cispd_orbitals
            gs_indices = list(range(len(orbitals)))   
            ex_indices = None                       
        else:
            # orbitals were passed, split them into ground state and excited state orbitals
            orbitals, ex_orbitals, gs_indices, ex_indices = self._split_orbitals(orbitals)

        # ground state orbitals (HF + MP2 PNOs)
        info = get_function_info(orbitals)
        nhf = len([x for x in info if numpy.isclose(float(x["occ"]), 2.0)])
        diagonal = {k: [] for k in range(nhf)}
        off_diagonal = {(k, l): [] for k in range(nhf) for l in range(k, nhf)}
        for k in range(len(orbitals)):
            x = info[k]["pair1"]
            y = info[k]["pair2"]
            orig_k = gs_indices[k]  
            if x == y:
                diagonal[x].append(orig_k)
            else:
                off_diagonal[(x, y)].append(orig_k)

        if ex_orbitals:
            # excited state orbitals (CIS X functions and CISPD PNOs)
            offset = len(orbitals)
            ex_info = get_function_info(ex_orbitals)

            def parse_label(label):
                # CIS_X_EX{ex}_X{idx}  oder  CISPD_EX{ex}
                if label.startswith("CIS_X_EX"):
                    return int(label.split("_")[2][2:]), "cis"
                elif label.startswith("CISPD_EX"):
                    return int(label.split("_")[-1][2:]), "cispd"
                return None, None

            def map_ex(k_ex):
                return ex_indices[k_ex] if ex_indices is not None else k_ex + offset
            
            ex_per_orb = {}
            for k_ex, x in enumerate(ex_info):
                ex, type = parse_label(x["type"])
                if type != "cis":
                    continue
                k_orb = int(x["pair1"]) 
                ex_per_orb.setdefault(k_orb, []).append(ex)

                if k_orb in diagonal:
                    diagonal[k_orb].append(map_ex(k_ex))
                print(f'excitations per orbital: {ex_per_orb}')

            for k_ex, x in enumerate(ex_info):
                ex, type = parse_label(x["type"])
                if type != "cispd":
                    continue
                xi, yi = int(x["pair1"]), int(x["pair2"])

                if xi == yi:
                    orb_excitations = ex_per_orb.get(xi)
                    if orb_excitations is not None and ex in orb_excitations:
                        diagonal[xi].append(map_ex(k_ex))
                else:
                    orb_excitations_x = ex_per_orb.get(xi, [])
                    orb_excitations_y = ex_per_orb.get(yi, [])
                    if ex in orb_excitations_x or ex in orb_excitations_y:
                        key = (min(xi, yi), max(xi, yi))
                        if key in off_diagonal:
                            off_diagonal[key].append(map_ex(k_ex))

            print(f'diagonal: {diagonal}')

        if use_diagonal:
            return diagonal
        return {**diagonal, **off_diagonal}

    def get_spa_edges(self, frozen_core=True, orbitals=None):
        pno_groupings = self.get_pno_groupings(diagonal=True, orbitals=orbitals)
        edges = [tuple(sorted(x)) for x in pno_groupings.values()]
        nfreeze = self.impl.get_frozen_core_dim()
        if frozen_core:
            orbitals = self.get_orbitals()
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
            "econv": 1.0e-4,
            "dconv": 5.0e-4,
            "localize": "boys",
        }
        data["nemo"] = {"ncf": "( none , 1.0)"}

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

        for item in ["dft", "pno", "nemo", "plot"]:
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
