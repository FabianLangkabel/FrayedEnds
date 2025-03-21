#include "integrals.hpp"

int main(int argc, char** argv)
{
    double box_size = 50;
    int wavelet_order = 7;
    double madness_thresh = 0.0001;

    Integrals* ints = new Integrals(argc, argv, box_size, wavelet_order, madness_thresh);
    ints->ReadOrbitals();
    ints->CalculateIntegrals();

    return 0;
}