from ._madpy_impl import NWChem_Converter as converter
from .madworld import redirect_output


class NWChem_Converter:

    _mos = None
    _normalized_aos = None
    impl = None

    @property
    def mos(self, *args, **kwargs):
        return self.get_mos(*args, **kwargs)

    @property
    def normalized_aos(self, *args, **kwargs):
        return self.get_normalized_aos(*args, **kwargs)

    def __init__(self, madworld, *args, **kwargs):
        self.impl = converter(madworld._impl)

    @redirect_output("read_nwchem_file.log")
    def read_nwchem_file(self, file, *args, **kwargs):
        self.impl.read_nwchem_file(file)

    def get_normalized_aos(self, *args, **kwargs):
        if self._normalized_aos is None:
            self._normalized_aos = self.impl.get_normalized_aos(*args, **kwargs)
            assert self._normalized_aos is not None
        return self._normalized_aos

    def get_mos(self, *args, **kwargs):
        if self._mos is None:
            self._mos = self.impl.get_mos(*args, **kwargs)
            assert self._mos is not None
        return self._mos

    def get_Vnuc(self):
        return self.impl.get_vnuc()

    def get_nuclear_repulsion_energy(self):
        return self.impl.get_nuclear_repulsion_energy()
