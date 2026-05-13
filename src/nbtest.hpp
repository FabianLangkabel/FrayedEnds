#pragma once

#include <nanobind/ndarray.h>

using namespace madness;
namespace nb = nanobind;

class NBArraytest {
    public:
        std::tuple<unsigned int, unsigned int> shape;
        Tensor<double> array;
        NBArraytest(std::tuple<unsigned int, unsigned int> shape) : shape(shape) {array = Tensor<double>(std::get<0>(shape), std::get<1>(shape));}

        void fill_array(std::vector<double> data){
            for (int i = 0; i < std::get<0>(shape); i++) {
                for (int j = 0; j < std::get<1>(shape); j++) {
                    array(i, j) = data[i * std::get<1>(shape) + j];
                }
            }
        }

        nb::ndarray<nb::numpy, double, nb::ndim<2>> to_numpy() {
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(array.ptr(), {std::get<0>(shape), std::get<1>(shape)});
        }

        void double_all(){
            for (int i = 0; i < std::get<0>(shape); i++) {
                for (int j = 0; j < std::get<1>(shape); j++) {
                    array(i, j) *= 2;
                }
            }
        }

        nb::ndarray<nb::numpy, double, nb::ndim<2>> get_unsafe() {
            Tensor<double> brother_array(std::get<0>(shape), std::get<1>(shape));
            for (int i = 0; i < std::get<0>(shape); i++) {
                for (int j = 0; j < std::get<1>(shape); j++) {
                    brother_array(i, j) = array(i, j);
                }
            }
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(brother_array.ptr(), {std::get<0>(shape), std::get<1>(shape)});
        }

        auto get_cast() {
            Tensor<double> brother_array(std::get<0>(shape), std::get<1>(shape));
            for (int i = 0; i < std::get<0>(shape); i++) {
                for (int j = 0; j < std::get<1>(shape); j++) {
                    brother_array(i, j) = array(i, j);
                }
            }

            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(brother_array.ptr(), {std::get<0>(shape), std::get<1>(shape)}).cast();
        }

        nb::ndarray<nb::numpy, double, nb::ndim<2>> get_capsule_simpl() {
            Tensor<double>* brother_array = new Tensor<double>(array);
            std::cout << (*brother_array)(0,0) << (*brother_array)(0,1) << (*brother_array)(2,3) << std::endl;
            nb::capsule brother_caps(
                brother_array,
                [](void *p) noexcept {
                    delete reinterpret_cast<madness::Tensor<double>*>(p);
                }
            );
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(brother_array->ptr(), {std::get<0>(shape), std::get<1>(shape)}, brother_caps);
        }

        nb::ndarray<nb::numpy, double, nb::ndim<2>> get_capsule_convoluded() {
            auto brother_array = std::make_shared<Tensor<double>>(array);
            std::cout << (*brother_array)(0,0) << (*brother_array)(0,1) << (*brother_array)(2,3) << std::endl;
            nb::capsule brother_caps(
                new std::shared_ptr<madness::Tensor<double>>(brother_array),
                [](void *p) noexcept {
                    delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
                }
            );
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(brother_array->ptr(), {std::get<0>(shape), std::get<1>(shape)}, brother_caps);
        }

        void explode() {
            this->array = Tensor<double>();
        }
};