from ._madpy_impl import MadMolecule as MadMoleculeImpl


class MadMolecule:
    impl = None

    def __init__(self, *args, **kwargs):
        self.impl = MadMoleculeImpl(*args, **kwargs)

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
