#pragma once

#include <madness/mra/mra.h>
#include <madness/mra/operator.h>


//the CoulombOperatorND class is a generalization of the CoulombOperator class from madness/mra/operator.h for 1 and 2 dimensions
//Applying a CoulombOperatorND<N> object to a function of the same dimension N results in the convolution of the function with the Coulomb kernel 1/r



using namespace madness;

template <std::size_t NDIM>
std::pair<Tensor<double>,Tensor<double>> gauss_fit_coulomb(double lo, double hi, double eps, bool fix_interval = false) {
    //original implementation in madness/mra/gfit.h (function bsh_fit)
	eps=eps/(4.0*constants::pi);
    double TT;
    double slo, shi;

    if (eps >= 1e-2) TT = 5;
    else if (eps >= 1e-4) TT = 10;
    else if (eps >= 1e-6) TT = 14;
    else if (eps >= 1e-8) TT = 18;
    else if (eps >= 1e-10) TT = 22;
    else if (eps >= 1e-12) TT = 26;
    else TT = 30;

    slo = log(eps / hi) - 1.0;

    shi = 0.5 * log(TT / (lo * lo));
    if (shi <= slo) throw "bsh_fit: logic error in slo,shi";

    // Resolution required for quadrature over s
    double h = 1.0 / (0.2 - .50 * log10(eps)); // was 0.5 was 0.47

    // Truncate the number of binary digits in h's mantissa
    // so that rounding does not occur when performing
    // manipulations to determine the quadrature points and
    // to limit the number of distinct values in case of
    // multiple precisions being used at the same time.
    h = floor(64.0 * h) / 64.0;

    // Round shi/lo up/down to an integral multiple of quadrature points
    shi = ceil(shi / h) * h;
    slo = floor(slo / h) * h;

    long npt = long((shi - slo) / h + 0.5);

    Tensor<double> coeff(npt), expnt(npt);

    for (int i = 0; i < npt; ++i) {
        double s = slo + h * (npt - i); // i+1
        coeff[i] = h * 2.0 / sqrt(constants::pi) * exp(s);
        coeff[i] = coeff[i]/(4.0*constants::pi);
        expnt[i] = exp(2.0 * s);
    }

#if ONE_TERM
    npt=1;
    double s=1.0;
    coeff[0]=1.0;
    expnt[0] = exp(2.0*s);
    coeff=coeff(Slice(0,0));
    expnt=expnt(Slice(0,0));
    print("only one term in gfit",s,coeff[0],expnt[0]);


#endif

    // Prune large exponents from the fit ... never necessary due to construction

    // Prune small exponents from Coulomb fit.  Evaluate a gaussian at
    // the range midpoint, and replace it there with the next most
    // diffuse gaussian.  Then examine the resulting error at the two
    // end points ... if this error is less than the desired
    // precision, can discard the diffuse gaussian.

    if (not fix_interval) {
        //		if (restrict_interval) {
        GFit<double, NDIM>::prune_small_coefficients(eps, lo, hi, coeff, expnt);
    }

    //in the original implementation there is a bunch of code here which gets executed when a variable nmom is larger than 0,
    //however nmom is always set to zero in the original implementation, so I removed it
    coeff.scale(4.0*constants::pi);
    return {coeff,expnt};
}

template <std::size_t NDIM>
SeparatedConvolution<double,NDIM>* CoulombOperatorNDPtr(World& world,
                                                       double lo,
                                                       double eps,
                                                       const array_of_bools<NDIM>& lattice_summed = FunctionDefaults<NDIM>::get_bc().is_periodic(),
                                                       int k=FunctionDefaults<NDIM>::get_k())
{
    const Tensor<double> &cell_width =
              FunctionDefaults<NDIM>::get_cell_width();
    double hi = cell_width.normf(); // Diagonal width of cell
    // Extend kernel range for lattice summation
    // N.B. if have periodic boundaries, extend range just in case will be using periodic domain
    const auto lattice_summed_any = lattice_summed.any();
    if (lattice_summed.any() || FunctionDefaults<NDIM>::get_bc().is_periodic_any()) {
        hi *= 100;
    }
    auto [coeffs, expnts]=gauss_fit_coulomb<NDIM>(lo,hi,eps);
    return new SeparatedConvolution<double,NDIM>(world,coeffs,expnts, lo, eps, lattice_summed, k);
}

template <std::size_t NDIM>
SeparatedConvolution<double,NDIM> CoulombOperatorND(World& world,
                                                       double lo,
                                                       double eps,
                                                       const array_of_bools<NDIM>& lattice_summed = FunctionDefaults<NDIM>::get_bc().is_periodic(),
                                                       int k=FunctionDefaults<NDIM>::get_k())
{
    const Tensor<double> &cell_width =
              FunctionDefaults<NDIM>::get_cell_width();
    double hi = cell_width.normf(); // Diagonal width of cell
    // Extend kernel range for lattice summation
    // N.B. if have periodic boundaries, extend range just in case will be using periodic domain
    const auto lattice_summed_any = lattice_summed.any();
    if (lattice_summed.any() || FunctionDefaults<NDIM>::get_bc().is_periodic_any()) {
        hi *= 100;
    }
    auto [coeffs, expnts]=gauss_fit_coulomb<NDIM>(lo,hi,eps);
    return SeparatedConvolution<double,NDIM>(world,coeffs,expnts, lo, eps, lattice_summed, k);
}