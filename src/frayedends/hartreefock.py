import json

import numpy as np

from ._frayedends_impl import HartreeFock as HartreeFockImpl
from .madworld import redirect_output


class HartreeFock:
    """
    MRA Hartree-Fock calculation with the MADNESS SCF solver (the solver of moldft).

    The calculation uses the wavelet order k and the box size L of the MadWorld, so the resulting
    orbitals can directly be used in all other FrayedEnds classes. The occupied orbitals are started
    either from user-provided guess orbitals (e.g. NWChem MOs translated into MRA with the
    NWChem_Converter) or, if no guess is given, from the default MADNESS AO guess.

    Further MADNESS "dft" parameters can be passed as keyword arguments, e.g.
    econv=1e-6, dconv=1e-5, maxiter=30, protocol=[1e-4, 1e-6], charge=1, spin_restricted=False, nopen=1,
    hfexalg="smallmem" (exchange algorithm with lower memory consumption).
    All available parameters are listed by `moldft --print_parameters`.
    The defaults that differ from moldft are:
        localize="canon"  canonical orbitals (needed to separate e.g. sigma and pi orbitals)
        save=False        no restartdata files are written
        protocol          [thresh] of the MadWorld if guess orbitals are given, otherwise [1e-4, thresh]
    """

    impl = None
    energy = None
    parameters = None

    def __init__(
        self,
        madworld,
        molecule,
        initial_orbitals=None,
        initial_orbital_energies=None,
        initial_beta_orbitals=None,
        initial_beta_orbital_energies=None,
        **kwargs,
    ):
        if madworld.dimensions != 3:
            raise ValueError(
                f"Hartree-Fock calculations only possible in 3 dimensions. MadWorld is initialized with {madworld.dimensions} dims."
            )

        defaults = madworld.get_function_defaults()
        k = defaults["k"]
        L = defaults["cell_width"] / 2
        thresh = defaults["thresh"]

        if initial_orbitals is not None or thresh >= 1.0e-4:
            protocol = [thresh]
        else:
            protocol = [1.0e-4, thresh]

        parameters = {"k": k, "l": L, "protocol": protocol, "localize": "canon", "save": False}
        for key, value in kwargs.items():
            parameters[key.lower()] = value

        # orbitals with a different k or box can not be combined with the functions of the MadWorld
        if parameters["k"] != k or abs(parameters["l"] - L) > 1.0e-10:
            raise ValueError(
                f"k and L of the Hartree-Fock calculation have to match the MadWorld (k={k}, L={L}), "
                f"got k={parameters['k']}, L={parameters['l']}."
            )
        self.parameters = parameters

        self.impl = HartreeFockImpl(madworld.impl, molecule.impl, json.dumps(parameters))

        if initial_orbitals is not None:
            self.set_initial_orbitals(
                initial_orbitals, initial_orbital_energies, initial_beta_orbitals, initial_beta_orbital_energies
            )

    def set_initial_orbitals(self, orbitals, orbital_energies=None, beta_orbitals=None, beta_orbital_energies=None):
        """
        Guess for the occupied orbitals (list of SavedFct3D). The orbital energies are optional,
        beta orbitals are only needed for spin-unrestricted calculations.
        """

        def as_list(values):
            return [] if values is None else [float(v) for v in values]

        self.impl.set_initial_orbitals(
            list(orbitals),
            as_list(orbital_energies),
            [] if beta_orbitals is None else list(beta_orbitals),
            as_list(beta_orbital_energies),
        )

    @redirect_output("hartree_fock.log")
    def solve(self):
        self.energy = self.impl.solve()
        return self.energy

    @property
    def orbitals(self):
        return self.get_orbitals()

    def get_orbitals(self):
        # occupied (alpha) orbitals
        return self.impl.get_alpha_orbitals()

    def get_beta_orbitals(self):
        return self.impl.get_beta_orbitals()

    def get_orbital_energies(self):
        return np.array(self.impl.get_alpha_orbital_energies())

    def get_beta_orbital_energies(self):
        return np.array(self.impl.get_beta_orbital_energies())

    def get_minimal_basis(self):
        # sto-3g AOs in MRA representation
        return self.impl.get_minimal_basis()

    def get_energy(self):
        return self.impl.get_energy()

    def get_vnuc(self):
        # nuclear potential of the calculation (includes the eprec of the molecule)
        return self.impl.get_vnuc()

    def get_nuclear_repulsion(self):
        return self.impl.get_nuclear_repulsion()
