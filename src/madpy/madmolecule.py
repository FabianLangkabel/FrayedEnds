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

    def get_vnuc(self, madworld):
        return self.impl.get_vnuc(madworld._impl)
