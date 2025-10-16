#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include "functionsaver.hpp"
#include "madness_process.hpp"
#include <nanobind/stl/function.h>

using namespace madness;
namespace nb = nanobind;

template <std::size_t NDIM> struct FunctionType;

template <> struct FunctionType<1> {
    using type = std::function<double(double)>;
};

template <> struct FunctionType<2> {
    using type = std::function<double(double, double)>;
};

template <> struct FunctionType<3> {
    using type = std::function<double(double, double, double)>;
};

// helper class to create a functor from a Python function
template <std::size_t NDIM> class PyFunctor : public FunctionFunctorInterface<double, NDIM> {
  public:
    using FuncType = typename FunctionType<NDIM>::type;
    FuncType f;

    PyFunctor(FuncType pyfunc) : f(pyfunc) {}

    double operator()(const Vector<double, NDIM>& r) const override {
        if constexpr (NDIM == 1) {
            return f(r[0]);
        } else if constexpr (NDIM == 2) {
            return f(r[0], r[1]);
        } else if constexpr (NDIM == 3) {
            return f(r[0], r[1], r[2]);
        }
    }
};

template <std::size_t NDIM> class PyFuncFactory {
  public:
    Function<double, NDIM> MRA_func;
    using FuncType = typename FunctionType<NDIM>::type;

    PyFuncFactory(MadnessProcess<NDIM>& mp, FuncType pyfunc) {
        // for this process n_threads always has to be 0, otherwise the python function can not be converted to a MRA
        // function
        std::cout.precision(6);

        // save the current number of threads and set it to 0
        std::size_t nthreads_at_start = ThreadPool::size();
        if (nthreads_at_start != 0) {
            ThreadPool::end();
            ThreadPool::begin(0);
            std::cout << "Changed number of threads to " << ThreadPool::size() << "." << std::endl;
        }

        PyFunctor<NDIM> functor(pyfunc);
        MRA_func = FunctionFactory<double, NDIM>(*(mp.world)).functor(functor);

        if (nthreads_at_start != 0) {
            ThreadPool::end();
            ThreadPool::begin(nthreads_at_start);
            std::cout << "Created MRA Function and changed number of threads back to " << ThreadPool::size() << "."
                      << std::endl;
        }
    }
    ~PyFuncFactory() { MRA_func.clear(); }
    SavedFct<NDIM> get_mra_function() { return SavedFct(MRA_func); }
};

template class PyFuncFactory<2>;
template class PyFuncFactory<3>;