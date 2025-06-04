#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include "functionsaver.hpp"
#include "MadnessProcess.hpp"
#include <nanobind/stl/function.h>

using namespace madness;
namespace nb = nanobind;

//helper class to create a functor from a Python function
class PyFunctor: public FunctionFunctorInterface<double, 3> {
    public:
        std::function<double(double, double, double)> f; 

        PyFunctor(std::function<double(double, double, double)> pyfunc): f(pyfunc) {}

        double operator()(const Vector<double,3> &r) const override {
            return f(r[0],r[1],r[2]);
        }
};

class PyFuncFactory{
    public:
        World* world;
        Function<double,3> MRA_func;

        PyFuncFactory(double L, long k, double thresh, std::function<double(double, double, double)> pyfunc) {
            int arg = 0;
            char **a = new char*[0]();

            world = &initialize(arg, a, 0);
            startup(*world,arg,a);
            delete[] a;
            
            FunctionDefaults<3>::set_k(k);
            FunctionDefaults<3>::set_thresh(thresh);
            FunctionDefaults<3>::set_refine(true);
            FunctionDefaults<3>::set_initial_level(5);
            FunctionDefaults<3>::set_truncate_mode(1);
            FunctionDefaults<3>::set_cubic_cell(-L, L);

            std::cout.precision(6);
            std::cout << "Creating function with functor" << std::endl;
            PyFunctor functor(pyfunc);
            std::cout << "Functor created" << std::endl;
            MRA_func = FunctionFactory<double,3>(*world).functor(functor);
            std::cout << "Function created" << std::endl;
        }
        ~PyFuncFactory(){
            MRA_func.clear();
            std::cout << "Finalize madness env" << std::endl;
            world->gop.fence();
            finalize();
        }
        SavedFct GetMRAFunction() {
            return SavedFct(MRA_func);
        }
};






