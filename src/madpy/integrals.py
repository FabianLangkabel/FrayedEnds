from tequila.quantumchemistry import NBodyTensor

from ._madpy_impl import Integrals2D as IntegralsInterface2D
from ._madpy_impl import Integrals3D as IntegralsInterface3D
from .mrafunctionwrapper import MRAFunction2D, MRAFunction3D


class Integrals3D:

    impl = None

    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface3D(madworld.impl)

    def compute_two_body_integrals(
        self,
        orbitals,
        ordering="phys",
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        g_elems = self.impl.compute_two_body_integrals(
            [orb.impl for orb in orbitals],
            truncation_tol,
            coulomb_lo,
            coulomb_eps,
            nocc,
        )
        g = NBodyTensor(elems=g_elems, ordering="phys")
        if ordering != "phys":
            return g.reorder(to=ordering)
        else:
            return g

    def compute_frozen_core_interaction(
        self,
        frozen_core_orbs,
        active_orbs,
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        return self.impl.compute_frozen_core_interaction(
            [fr_orb.impl for fr_orb in frozen_core_orbs],
            [act_orb.impl for act_orb in active_orbs],
            truncation_tol,
            coulomb_lo,
            coulomb_eps,
            nocc,
        )

    def compute_kinetic_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_kinetic_integrals([orb.impl for orb in orbitals])

    def compute_potential_integrals(self, orbitals, V, *args, **kwargs):
        return self.impl.compute_potential_integrals(
            [orb.impl for orb in orbitals], V.impl
        )

    def compute_overlap_integrals(self, orbitals, other=None, *args, **kwargs):
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(
            [orb.impl for orb in orbitals], [other_orb.impl for other_orb in other]
        )

    def orthonormalize(
        self, orbitals, method="symmetric", rr_thresh=0.0, *args, **kwargs
    ):
        orth_orbs_impl = self.impl.orthonormalize(
            [orb.impl for orb in orbitals], method, rr_thresh
        )
        return [
            MRAFunction3D(
                orth_orbs_impl[i], type=orbitals[i].type, info=orbitals[i].info
            )
            for i in range(len(orth_orbs_impl))
        ]

    def project_out(self, kernel, target, *args, **kwargs):
        res_target_impl = self.impl.project_out(
            [k.impl for k in kernel], [t.impl for t in target]
        )
        return [
            MRAFunction3D(res_target_impl[i], type=target[i].type, info=target[i].info)
            for i in range(len(target))
        ]

    def project_on(self, kernel, target, *args, **kwargs):
        res_target_impl = self.impl.project_on(
            [k.impl for k in kernel], [t.impl for t in target]
        )
        return [
            MRAFunction3D(res_target_impl[i], type=target[i].type, info=target[i].info)
            for i in range(len(target))
        ]

    def normalize(self, orbitals, *args, **kwargs):
        norm_orbs_impl = self.impl.normalize(
            [orb.impl for orb in orbitals], *args, **kwargs
        )
        return [
            MRAFunction3D(
                norm_orbs_impl[i], type=orbitals[i].type, info=orbitals[i].info
            )
            for i in range(len(orbitals))
        ]

    def transform(self, orbitals, matrix, *args, **kwargs):
        trans_orbs_impl = self.impl.transform([orb.impl for orb in orbitals], matrix)
        return [
            MRAFunction3D(
                trans_orbs_impl[i],
                type=orbitals[i].type,
                info=orbitals[i].info + " transformed",
            )
            for i in range(len(orbitals))
        ]

class Integrals2D:

    impl = None

    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface2D(madworld.impl)

    def compute_two_body_integrals(
        self,
        orbitals,
        ordering="phys",
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        g_elems = self.impl.compute_two_body_integrals(
            [orb.impl for orb in orbitals],
            truncation_tol,
            coulomb_lo,
            coulomb_eps,
            nocc,
        )
        g = NBodyTensor(elems=g_elems, ordering="phys")
        if ordering != "phys":
            return g.reorder(to=ordering)
        else:
            return g

    def compute_frozen_core_interaction(
        self,
        frozen_core_orbs,
        active_orbs,
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        return self.impl.compute_frozen_core_interaction(
            [fr_orb.impl for fr_orb in frozen_core_orbs],
            [act_orb.impl for act_orb in active_orbs],
            truncation_tol,
            coulomb_lo,
            coulomb_eps,
            nocc,
        )

    def compute_kinetic_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_kinetic_integrals([orb.impl for orb in orbitals])

    def compute_potential_integrals(self, orbitals, V, *args, **kwargs):
        return self.impl.compute_potential_integrals(
            [orb.impl for orb in orbitals], V.impl
        )

    def compute_overlap_integrals(self, orbitals, other=None, *args, **kwargs):
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(
            [orb.impl for orb in orbitals], [other_orb.impl for other_orb in other]
        )

    def orthonormalize(
        self, orbitals, method="symmetric", rr_thresh=0.0, *args, **kwargs
    ):
        orth_orbs_impl = self.impl.orthonormalize(
            [orb.impl for orb in orbitals], method, rr_thresh
        )
        return [
            MRAFunction2D(
                orth_orbs_impl[i], type=orbitals[i].type, info=orbitals[i].info
            )
            for i in range(len(orth_orbs_impl))
        ]

    def project_out(self, kernel, target, *args, **kwargs):
        res_target_impl = self.impl.project_out(
            [k.impl for k in kernel], [t.impl for t in target]
        )
        return [
            MRAFunction2D(res_target_impl[i], type=target[i].type, info=target[i].info)
            for i in range(len(target))
        ]

    def project_on(self, kernel, target, *args, **kwargs):
        res_target_impl = self.impl.project_on(
            [k.impl for k in kernel], [t.impl for t in target]
        )
        return [
            MRAFunction2D(res_target_impl[i], type=target[i].type, info=target[i].info)
            for i in range(len(target))
        ]

    def normalize(self, orbitals, *args, **kwargs):
        norm_orbs_impl = self.impl.normalize(
            [orb.impl for orb in orbitals], *args, **kwargs
        )
        return [
            MRAFunction2D(
                norm_orbs_impl[i], type=orbitals[i].type, info=orbitals[i].info
            )
            for i in range(len(orbitals))
        ]

    def transform(self, orbitals, matrix, *args, **kwargs):
        trans_orbs_impl = self.impl.transform([orb.impl for orb in orbitals], matrix)
        return [
            MRAFunction2D(
                trans_orbs_impl[i],
                type=orbitals[i].type,
                info=orbitals[i].info + " transformed",
            )
            for i in range(len(orbitals))
        ]
