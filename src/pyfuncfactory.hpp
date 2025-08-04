#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include "functionsaver.hpp"
#include "madness_process.hpp"
#include <nanobind/stl/function.h>

using namespace madness;
namespace nb = nanobind;

// helper class to create a functor from a Python function
class PyFunctor : public FunctionFunctorInterface<double, 3> {
  public:
    std::function<double(double, double, double)> f;

    PyFunctor(std::function<double(double, double, double)> pyfunc) : f(pyfunc) {}

    double operator()(const Vector<double, 3>& r) const override { return f(r[0], r[1], r[2]); }
};

class PyFuncFactory {
  public:
    Function<double, 3> MRA_func;

    PyFuncFactory(MadnessProcess& mp, std::function<double(double, double, double)> pyfunc) {
        // for this process n_threads always has to be 0, otherwise the python function can not be converted to a MRA
        // function
        std::cout.precision(6);

        // save the current number of threads and set it to 0
        std::size_t nthreads_at_start = ThreadPool::size();
        if (nthreads_at_start != 0){
            ThreadPool::end();
            ThreadPool::begin(0);
            std::cout << "Changed number of threads to " << ThreadPool::size() << "." << std::endl;
        }

        PyFunctor functor(pyfunc);
        MRA_func = FunctionFactory<double, 3>(*(mp.world)).functor(functor);
        

        if (nthreads_at_start != 0){
            ThreadPool::end();
            ThreadPool::begin(nthreads_at_start);
            std::cout << "Created MRA Function and changed number of threads back to " << ThreadPool::size() << "." << std::endl;
        }
    }
    ~PyFuncFactory() { MRA_func.clear(); }
    SavedFct get_mra_function() { return SavedFct(MRA_func); }
};
