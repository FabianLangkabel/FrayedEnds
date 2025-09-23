from ._madpy_impl import NWChem_Converter_open_shell as converter_open_shell
from .madworld import redirect_output


class NWChem_Converter_open_shell:

    _alpha_mos = None
    _beta_mos = None
    _normalized_aos = None
    impl = None

    @property
    def mos(self, *args, **kwargs):
        return self.get_mos(*args, **kwargs)

    @property
    def normalized_aos(self, *args, **kwargs):
        return self.get_normalized_aos(*args, **kwargs)

    def __init__(self, madworld, *args, **kwargs):
        self.impl = converter_open_shell(madworld._impl)

    @redirect_output("read_nwchem_file.log")
    def read_nwchem_file(self, file, *args, **kwargs):
        self.impl.read_nwchem_file(file)

    def get_normalized_aos(self, *args, **kwargs):
        if self._normalized_aos is None:
            self._normalized_aos = self.impl.get_normalized_aos(*args, **kwargs)
            assert self._normalized_aos is not None
        return self._normalized_aos

    def get_mos(self, *args, **kwargs):
        if self._alpha_mos is None:
            self._alpha_mos = self.impl.get_alpha_mos(*args, **kwargs)
            assert self._alpha_mos is not None
        if self._beta_mos is None:
            self._beta_mos = self.impl.get_beta_mos(*args, **kwargs)
            assert self._beta_mos is not None
        return [self._alpha_mos, self._beta_mos]

    def get_Vnuc(self):
        return self.impl.get_vnuc()

    def get_nuclear_repulsion_energy(self):
        return self.impl.get_nuclear_repulsion_energy()
