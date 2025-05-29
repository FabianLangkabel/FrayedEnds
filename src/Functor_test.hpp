#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include "functionsaver.hpp"
#include "MadnessProcess.hpp"
#include <nanobind/stl/function.h>

using namespace madness;
namespace nb = nanobind;

class Functor: public FunctionFunctorInterface<double, 1> {
    public:
        std::function<double(double)> f; 

        Functor(std::function<double(double)> f): f(f) {}

        double operator()(const Vector<double,1> &r) const override {
            return f(r[0]);
        }
};

class CreateFunc{
    public:
        World* world;
        Function<double,1> f1;

        CreateFunc(double L, long k, double thresh, std::function<double(double)> f) {
            int arg = 0;
            char **a = new char*[0]();

            world = &initialize(arg, a, 1);
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
            Functor functor(f);
            std::cout << "Functor created" << std::endl;
            f1 = FunctionFactory<double,1>(*world).functor(functor);
            std::cout << "Function created" << std::endl;
        }
        ~CreateFunc(){
            f1.clear();
            std::cout << "Finalize madness env" << std::endl;
            world->gop.fence();
            finalize();
        }
};

double f(double x) {
    return std::sin(x);
}
std::function<double(double)> return_f(){
    return f;
}



double call_python_function(std::function<double(double)> python_func, double x) {
    // Call the Python function
    return python_func(x);
}



/*
class CreateFunct: public MadnessProcess{
    CreateFunct(double L, long k, double thresh): MadnessProcess(L,k,thresh) {
        std::cout.precision(6);
        Function<double,1> f1 = FunctionFactory<double,1>(*world).functor(functor).special_level(initial_level).special_points({Vector<double,1>(-L), Vector<double,1>(L)}).truncate_mode(truncate_mode).refine(refine);
        SavedFct sf(f1);
        write_binary_file(sf,"saved_fct");
    }
}
*/

double test_function(Functor F, const double x) {
    return F(Vector<double,1>(x));
}

