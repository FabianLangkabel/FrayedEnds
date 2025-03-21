#include <nanobind/nanobind.h>
#include "integrals.hpp"

namespace nb = nanobind;
/*
int add(int a, int b) { return a + b; }
NB_MODULE(MRA_integrals, m) {
    m.def("add", &add);
}
*/

NB_MODULE(MRA_integrals, m) {
    nb::class_<Integrals>(m, "Integrator")
        .def(nb::init<const double &, const int &, const double &>())
        .def("ReadOrbitals", &Integrals::ReadOrbitals)
        .def("CalculateIntegrals", &Integrals::CalculateIntegrals);
}