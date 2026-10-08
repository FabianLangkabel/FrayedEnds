import os
import re
import shlex
import shutil
import subprocess

from ._frayedends_impl import NWChem_Converter as converter
from ._frayedends_impl import NWChem_Converter_open_shell as converter_open_shell
from .madworld import redirect_output
from .moleculargeometry import MolecularGeometry


def run_nwchem(
    geometry: str,
    basis: str,
    name: str = "nwchem",
    workdir: str = "nwchem",
    units: str = "angstrom",
    charge: int = 0,
    multiplicity: int = 1,
    scf_thresh: float = 1.0e-8,
    maxiter: int = 200,
    memory_mb: int = 2000,
    spherical: bool = True,
    direct: bool = True,
    nwchem_command: str = "nwchem",
    extra_input: str = "",
    silent: bool = False,
) -> str:
    """
    Runs an NWChem SCF calculation and returns the file prefix (workdir/name), which can be passed to
    NWChem_Converter.read_nwchem_file / NWChem_Converter_open_shell.read_nwchem_file.

    geometry: one atom per line, "symbol x y z"
    multiplicity: 1 -> RHF, otherwise UHF with multiplicity-1 unpaired electrons
    nwchem_command: e.g. "nwchem" or "mpirun -np 4 nwchem"
    extra_input: additional NWChem input, inserted before "task scf"

    The geometry is not centered, reoriented or symmetrized by NWChem (noautoz nocenter noautosym),
    so the NWChem orbitals live in the coordinate frame of the given geometry.
    """
    os.makedirs(workdir, exist_ok=True)
    workdir = os.path.abspath(workdir)

    atoms = "\n".join("  " + line.strip() for line in geometry.strip().split("\n") if line.strip())
    if multiplicity == 1:
        scf_type = "rhf\n  singlet"
    else:
        scf_type = f"uhf\n  nopen {multiplicity - 1}"

    nwchem_input = f"""start {name}
title "{name}"
permanent_dir {workdir}
scratch_dir {workdir}
memory total {memory_mb} mb
charge {charge}
geometry units {units} noautoz nocenter noautosym
  symmetry c1
{atoms}
end
basis {"spherical" if spherical else "cartesian"}
  * library {basis}
end
scf
  {scf_type}
  thresh {scf_thresh}
  maxiter {maxiter}
  {"direct" if direct else ""}
end
{extra_input}
task scf
"""
    with open(os.path.join(workdir, name + ".nw"), "w") as f:
        f.write(nwchem_input)

    # the basis set library is usually set by "conda activate", derive it from the executable if it is missing
    env = os.environ.copy()
    executable = shutil.which(shlex.split(nwchem_command)[-1])
    if "NWCHEM_BASIS_LIBRARY" not in env and executable is not None:
        share = os.path.join(os.path.dirname(os.path.realpath(executable)), "..", "share", "nwchem")
        if os.path.isdir(os.path.join(share, "libraries")):
            env["NWCHEM_BASIS_LIBRARY"] = os.path.abspath(os.path.join(share, "libraries")) + "/"
            env.setdefault("NWCHEM_NWPW_LIBRARY", os.path.abspath(os.path.join(share, "libraryps")) + "/")

    if not silent:
        print(f"Running NWChem ({basis}) in {workdir} ...")
    with open(os.path.join(workdir, name + ".out"), "w") as out, open(os.path.join(workdir, name + ".err"), "w") as err:
        process = subprocess.run(
            shlex.split(nwchem_command) + [name + ".nw"], cwd=workdir, stdout=out, stderr=err, env=env
        )

    with open(os.path.join(workdir, name + ".out")) as f:
        output = f.read()
    energies = re.findall(r"Total SCF energy =\s+(-?\d+\.\d+)", output)
    if process.returncode != 0 or not energies:
        raise RuntimeError(
            f"NWChem calculation failed (return code {process.returncode}), see {os.path.join(workdir, name + '.out')}"
        )
    if not silent:
        print(f"NWChem SCF energy: {float(energies[-1]):.10f}")

    return os.path.join(workdir, name)


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
        if madworld.dimensions != 3:
            raise ValueError(
                f"NWChem conversion only possible in 3 dimensions. MadWorld is initialized with {madworld.dimensions} dims."
            )
        self.impl = converter(madworld.impl)

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

    def get_occupancies(self):
        # occupation numbers of the NWChem MOs (2 or 0 for RHF)
        return self.impl.get_occupancies()

    def get_orbital_energies(self):
        return self.impl.get_orbital_energies()

    def get_molecular_geometry(self, eprec=None):
        # molecule in the coordinate frame of the NWChem calculation (units: bohr)
        molecule = MolecularGeometry(units="bohr", silent=True, eprec=eprec)
        for symbol, x, y, z in self.impl.get_atoms():
            molecule.add_atom(x, y, z, symbol)
        return molecule


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
        self.impl = converter_open_shell(madworld.impl)

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
