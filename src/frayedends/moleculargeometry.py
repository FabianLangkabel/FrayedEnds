import re
import json

from tequila import Molecule
from pyscf.gto import M
from scipy.constants import physical_constants

from ._frayedends_impl import MolecularGeometry as MolecularGeometryImpl
from .integrals import Integrals3D

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
            elif units in ["bohr", "atomic", "atomic units", "au", "a.u."]:
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
            geometry = re.sub(r" +", " ", geometry)
            print(geometry)
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
        json_str = self.impl.to_json()
        return json.loads(json_str)
    
    def get_geometry_string(self):
        c_bohrtoang = physical_constants["Bohr radius"][0] * 10**10
        coords_in_bohr = self.to_json()["geometry"]
        atomic_symbols = self.to_json()["symbols"]
        if self.impl.units == "angstrom":
            coord_in_ang = []
            for coord in coords_in_bohr:
                coord_in_ang.append(
                    [coord[0] * c_bohrtoang, coord[1] * c_bohrtoang, coord[2] * c_bohrtoang]
                )
            geom_str = ""
            for i in range(len(atomic_symbols)):
                geom_str += f"{atomic_symbols[i]} {coord_in_ang[i][0]} {coord_in_ang[i][1]} {coord_in_ang[i][2]}\n"
            return (geom_str, "angstrom")
        else:
            geom_str = ""
            for i in range(len(atomic_symbols)):
                geom_str += f"{atomic_symbols[i]} {coords_in_bohr[i][0]} {coords_in_bohr[i][1]} {coords_in_bohr[i][2]}\n"
            return (geom_str, "bohr")

    def molecular_potential_derivative(self, madworld, atom, axis):
        return self.impl.molecular_potential_derivative(madworld.impl, atom, axis)

    def molecular_potential_second_derivative(
        self, madworld, atom: int, axis1: int, axis2: int
    ):
        return self.impl.molecular_potential_second_derivative(
            madworld.impl, atom, axis1, axis2
        )

    def nuclear_repulsion_derivative(self, atom: int, axis: int):
        return self.impl.nuclear_repulsion_derivative(atom, axis)

    def nuclear_repulsion_second_derivative(
        self, atom1: int, atom2: int, axis1: int, axis2: int
    ):
        return self.impl.nuclear_repulsion_second_derivative(atom1, atom2, axis1, axis2)

    def compute_energy_gradient(self, madworld, orbitals, rdm1, nocc=2):
        # function to compute the energy gradient w.r.t. nuclear coordinates
        # this function assumes that the Hellmann-Feynmann theorem holds, 
        # i. e. that the partial derivate of the energy functional w.r.t. the orbitals or many-body wave function is zero
        n_atoms = len(self.to_json()["symbols"])
        fr_core_orbs=[]
        act_orbs=[]
        for orb in orbitals:
            if orb.type=="frozen_occ":
                fr_core_orbs.append(orb)
            else:
                act_orbs.append(orb)
        
        if len(act_orbs)!=rdm1.shape[0]:
            raise ValueError("Number of active orbitals does not match 1-RDM size. Specify which orbitals are active and frozen by setting 'type' attribute to 'active' or 'frozen_occ'")

        integrals = Integrals3D(madworld)
        gradV = []
        
        for atom in range(n_atoms):
            gradV_atom = []
            for axis in range(3):
                derivV = self.molecular_potential_derivative(madworld, atom, axis)
                val = 0.0
                for i in range(len(fr_core_orbs)):
                    fc_orb_derivV_fc_orb = integrals.compute_potential_integrals([fr_core_orbs[i]], derivV)
                    val += nocc*fc_orb_derivV_fc_orb[0,0]
                
                a_orbs_derivV_a_orbs = integrals.compute_potential_integrals(act_orbs, derivV)
                for i in range(len(act_orbs)):
                    for j in range(len(act_orbs)):
                        val += rdm1[i, j] * a_orbs_derivV_a_orbs[i, j]
                
                val += self.nuclear_repulsion_derivative(atom, axis)
                gradV_atom.append(val)
                
            gradV.append(gradV_atom)
        
        return gradV
        
    def get_vnuc(self, madworld):
        return self.impl.get_vnuc(madworld.impl)

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
        return Molecule(self.get_geometry_string()[0], units = self.impl.units, *args, **kwargs)

    # conversion from pyscf Mole to molecular geometry
    def from_pyscf_mol(pyscf_mol, units="angstrom"):
        if units in ["bohr", "atomic units", "au", "a.u."]:
            coords = pyscf_mol.atom_coords()
            output = []
            for i in range(pyscf_mol.natm):
                symb = pyscf_mol.atom_pure_symbol(i)
                x, y, z = coords[i]
                output.append('%-4s %17.8f %17.8f %17.8f' %
                          (symb, x, y, z))
            return MolecularGeometry(geometry="\n".join(output), units='bohr')
        else:
            return MolecularGeometry(geometry=pyscf_mol.tostring(), units=units)
    
    # conversion from molecular geometry to pyscf Mole
    def to_pyscf_mol(self, *args, **kwargs):
        return M(atom=self.get_geometry_string()[0], unit=self.impl.units, *args, **kwargs)
        