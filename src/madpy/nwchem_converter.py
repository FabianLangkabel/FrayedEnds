from ._madpy_impl import NWChem_Converter as converter
from .madworld import redirect_output
from .mrafunctionwrapper import MRAFunction3D


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
        self.impl = converter(madworld.impl)

    @redirect_output("read_nwchem_file.log")
    def read_nwchem_file(self, file, *args, **kwargs):
        self.impl.read_nwchem_file(file)

    def get_normalized_aos(self, *args, **kwargs):
        if self._normalized_aos is None:
            normalized_aos_impl = self.impl.get_normalized_aos(*args, **kwargs)
            self._normalized_aos = [
                MRAFunction3D(ao, type="ao") for ao in normalized_aos_impl
            ]
            assert self._normalized_aos is not None
        return self._normalized_aos

    def get_mos(self, *args, **kwargs):
        if self._mos is None:
            mos_impl = self.impl.get_mos()
            self._mos = [MRAFunction3D(mo, type="mo") for mo in mos_impl]
            assert self._mos is not None
        return self._mos

    def get_Vnuc(self):
        return MRAFunction3D(self.impl.get_vnuc())

    def get_nuclear_repulsion_energy(self):
        return self.impl.get_nuclear_repulsion_energy()
