import re

from tequila import Molecule

from ._madpy_impl import MolecularGeometry as MolecularGeometryImpl
from .mrafunctionwrapper import MRAFunction3D


class MolecularGeometry:
    impl = None
    silent = False

    def __init__(self, geometry: str = None, units=None, silent=False, *args, **kwargs):
        self.silent = silent

        if units is None:
            if not self.silent:
                print(
                    "Warning: No units passed with geometry, assuming units are angstrom."
                )
            units = "angstrom"
        else:
            units = units.lower()
            if units in ["angstrom", "ang", "a", "å"]:
                units = "angstrom"
            elif units in ["bohr", "atomic units", "au", "a.u."]:
                units = "bohr"
            else:
                if not self.silent:
                    print(
                        "Warning: Units passed with geometry not recognized (available units are angstrom or bohr), assuming units are angstrom."
                    )
                units = "angstrom"

        self.impl = MolecularGeometryImpl(units)
        if geometry is not None:
            geometry = geometry.lower()
            geometry = geometry.strip()
            # Replace tabs with spaces
            geometry = geometry.replace("\t", " ")
            # Replace multiple whitespace characters with a single space
            re.sub(r"\s+", " ", geometry).strip()

            for line in geometry.split("\n"):
                data = line.split(" ")
                x = eval(data[1])
                y = eval(data[2])
                z = eval(data[3])
                s = data[0]
                self.add_atom(x, y, z, s)

    def check_units(self):
        return self.impl.units

    def add_atom(self, pos_x, pos_y, pos_z, symbol):
        self.impl.add_atom(pos_x, pos_y, pos_z, symbol)

    def to_json(self):
        return self.impl.to_json()

    def compute_nuclear_derivative(self, madworld, atom, axis):
        return MRAFunction3D(
            self.impl.compute_nuclear_derivative(madworld.impl, atom, axis)
        )

    def compute_second_nuclear_derivative(
        self, madworld, atom: int, axis1: int, axis2: int
    ):
        return MRAFunction3D(
            self.impl.compute_second_nuclear_derivative(
                madworld.impl, atom, axis1, axis2
            )
        )

    def nuclear_repulsion_derivative(self, atom: int, axis: int):
        return self.impl.nuclear_repulsion_derivative(atom, axis)

    def nuclear_repulsion_second_derivative(
        self, atom1: int, atom2: int, axis1: int, axis2: int
    ):
        return self.impl.nuclear_repulsion_second_derivative(atom1, atom2, axis1, axis2)

    def get_vnuc(self, madworld):
        return MRAFunction3D(self.impl.get_vnuc(madworld.impl))

    def get_nuclear_charge(self):
        return self.impl.get_nuclear_charge()

    def get_nuclear_repulsion(self):
        return self.impl.get_nuclear_repulsion()

    @property
    def n_electrons(self):
        return int(self.get_nuclear_charge())

    @property
    def n_core_electrons(self):
        return self.impl.get_core_n_electrons()

    # conversion from tequila molecule to molecular geometry
    def from_tq_mol(tq_mol, units="angstrom"):
        geometry = tq_mol.parameters.get_geometry_string(desired_units=units)
        return MolecularGeometry(geometry, units=units)

    # conversion from molecular geometry to tequila molecule
    def to_tq_mol(self, *args, **kwargs):
        pass
