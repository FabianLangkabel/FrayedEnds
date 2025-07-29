from ._madpy_impl import MadMolecule as MadMoleculeImpl
import re

class MadMolecule:
    impl = None

    def __init__(self, geometry:str=None, *args, **kwargs):
        self.impl = MadMoleculeImpl(*args, **kwargs)
        if geometry is not None:
            geometry = geometry.lower()
            geometry = geometry.strip()
            # Replace tabs with spaces
            geometry = geometry.replace('\t', ' ')
            # Replace multiple whitespace characters with a single space
            geometry
            re.sub(r'\s+', ' ', geometry).strip()

            for line in geometry.split("\n"):
                data = line.split(" ")
                x = eval(data[1])
                y = eval(data[2])
                z = eval(data[3])
                s = data[0]
                self.add_atom(x, y, z, s)

    def add_atom(self, pos_x, pos_y, pos_z, symbol):
        self.impl.add_atom(pos_x, pos_y, pos_z, symbol)

    def to_json(self):
        return self.impl.to_json()

    def compute_nuclear_derivative(self, madworld, atom, axis):
        return self.impl.compute_nuclear_derivative(madworld._impl, atom, axis)
    
    def compute_second_nuclear_derivative(self,madworld, atom: int, axis1: int, axis2: int):
        return self.impl.compute_second_nuclear_derivative(madworld._impl, atom, axis1, axis2)

    def nuclear_repulsion_derivative(self, atom: int, axis: int):
        return self.impl.nuclear_repulsion_derivative(atom, axis)
        
    def nuclear_repulsion_second_derivative(self, atom1: int, atom2: int, axis1: int, axis2: int):
        return self.impl.nuclear_repulsion_second_derivative(atom1, atom2, axis1, axis2)


    def get_vnuc(self, madworld):
        return self.impl.get_vnuc(madworld._impl)

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
