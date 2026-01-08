#pragma once

#include <cstddef>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <stdexcept>

using namespace madness;
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

namespace open_shell_utils {

    inline madness::Tensor<double> to_madness(const Numpy2D& arr) 
    {
        const auto n0 = arr.shape(0);
        const auto n1 = arr.shape(1);

        madness::Tensor<double> T(n0, n1);

        for (std::size_t i = 0; i < n0; ++i)
            for (std::size_t j = 0; j < n1; ++j)
                T(i, j) = arr(i, j);

        return T;
    }


    inline madness::Tensor<double> to_madness(const Numpy4D& arr) 
    {
        const auto s0 = arr.shape(0);
        const auto s1 = arr.shape(1);
        const auto s2 = arr.shape(2);
        const auto s3 = arr.shape(3);

        madness::Tensor<double> T(s0, s1, s2, s3);

        for (std::size_t i = 0; i < s0; ++i)
            for (std::size_t j = 0; j < s1; ++j)
                for (std::size_t k = 0; k < s2; ++k)
                    for (std::size_t l = 0; l < s3; ++l)
                        T(i, j, k, l) = arr(i, j, k, l);

        return T;
    }

    inline void sort_eigenpairs_descending(
        madness::Tensor<double>& eigenvectors,
        madness::Tensor<double>& eigenvalues)
    {
        const std::size_t n = eigenvalues.dim(0);

        std::vector<std::pair<double, std::size_t>> pairs;
        pairs.reserve(n);

        for (std::size_t i = 0; i < n; ++i)
            pairs.emplace_back(eigenvalues(i), i);

        std::sort(
            pairs.begin(),
            pairs.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

        madness::Tensor<double> sorted_eigenvalues(n);
        madness::Tensor<double> sorted_eigenvectors(n, n);

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t orig_idx = pairs[i].second;
            sorted_eigenvalues(i) = eigenvalues(orig_idx);

            for (std::size_t j = 0; j < n; ++j)
                sorted_eigenvectors(j, i) = eigenvectors(j, orig_idx);
        }

        eigenvalues  = sorted_eigenvalues;
        eigenvectors = sorted_eigenvectors;
    }


    inline madness::Tensor<double> matmul_mxm(
        const madness::Tensor<double>& A,
        const madness::Tensor<double>& B,
        std::size_t n)
    {
        madness::Tensor<double> C(n, n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < n; ++k)
                    sum += A(i, k) * B(k, j);
                C(i, j) = sum;
            }
        }

        return C;
    }

    inline void TransformMatrix(
        madness::Tensor<double>* ObjectMatrix,
        madness::Tensor<double>& TransformationMatrix)
    {
        const int n = TransformationMatrix.dim(0);
        madness::Tensor<double> temp = matmul_mxm(*ObjectMatrix, TransformationMatrix, n);
        *ObjectMatrix = matmul_mxm(transpose(TransformationMatrix), temp, n);
    }

    inline void TransformTensor(
        madness::Tensor<double>& ObjectTensor,
        madness::Tensor<double>& TransformationMatrix)
    {
        const int n = TransformationMatrix.dim(0);

        madness::Tensor<double> temp1(n, n, n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l = 0; l < n; l++) {
                        double k_value = 0.0;
                        for (int k = 0; k < n; k++)
                            k_value += TransformationMatrix(k, k2) * ObjectTensor(i, j, k, l);
                        temp1(i, j, k2, l) = k_value;
                    }

        madness::Tensor<double> temp2(n, n, n, n);
        for (int i2 = 0; i2 < n; i2++)
            for (int j = 0; j < n; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l = 0; l < n; l++) {
                        double i_value = 0.0;
                        for (int i = 0; i < n; i++)
                            i_value += TransformationMatrix(i, i2) * temp1(i, j, k2, l);
                        temp2(i2, j, k2, l) = i_value;
                    }

        madness::Tensor<double> temp3(n, n, n, n);
        for (int i2 = 0; i2 < n; i2++)
            for (int j = 0; j < n; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l2 = 0; l2 < n; l2++) {
                        double l_value = 0.0;
                        for (int l = 0; l < n; l++)
                            l_value += TransformationMatrix(l, l2) * temp2(i2, j, k2, l);
                        temp3(i2, j, k2, l2) = l_value;
                    }

        madness::Tensor<double> temp4(n, n, n, n);
        for (int i2 = 0; i2 < n; i2++)
            for (int j2 = 0; j2 < n; j2++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l2 = 0; l2 < n; l2++) {
                        double j_value = 0.0;
                        for (int j = 0; j < n; j++)
                            j_value += TransformationMatrix(j, j2) * temp3(i2, j, k2, l2);
                        temp4(i2, j2, k2, l2) = j_value;
                    }

        ObjectTensor = temp4;
    }

    inline void Transform_ab_mixed_Tensor(
        madness::Tensor<double>& ObjectTensor,
        madness::Tensor<double>& TransformationMatrix_alpha,
        madness::Tensor<double>& TransformationMatrix_beta)
    {
        const int n = TransformationMatrix_alpha.dim(0);
        const int m = TransformationMatrix_beta.dim(0);

        madness::Tensor<double> temp1(n, m, n, m);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l = 0; l < m; l++) {
                        double k_value = 0.0;
                        for (int k = 0; k < n; k++)
                            k_value += TransformationMatrix_alpha(k, k2) * ObjectTensor(i, j, k, l);
                        temp1(i, j, k2, l) = k_value;
                    }

        madness::Tensor<double> temp2(n, m, n, m);
        for (int i2 = 0; i2 < n; i2++)
            for (int j = 0; j < m; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l = 0; l < m; l++) {
                        double i_value = 0.0;
                        for (int i = 0; i < n; i++)
                            i_value += TransformationMatrix_alpha(i, i2) * temp1(i, j, k2, l);
                        temp2(i2, j, k2, l) = i_value;
                    }

        madness::Tensor<double> temp3(n, m, n, m);
        for (int i2 = 0; i2 < n; i2++)
            for (int j = 0; j < m; j++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l2 = 0; l2 < m; l2++) {
                        double l_value = 0.0;
                        for (int l = 0; l < m; l++)
                            l_value += TransformationMatrix_beta(l, l2) * temp2(i2, j, k2, l);
                        temp3(i2, j, k2, l2) = l_value;
                    }

        madness::Tensor<double> temp4(n, m, n, m);
        for (int i2 = 0; i2 < n; i2++)
            for (int j2 = 0; j2 < m; j2++)
                for (int k2 = 0; k2 < n; k2++)
                    for (int l2 = 0; l2 < m; l2++) {
                        double j_value = 0.0;
                        for (int j = 0; j < m; j++)
                            j_value += TransformationMatrix_beta(j, j2) * temp3(i2, j, k2, l2);
                        temp4(i2, j2, k2, l2) = j_value;
                    }

        ObjectTensor = temp4;
    }

    // ============================================================
    // Generic contraction utilities
    // ============================================================

    // Compile-time nested loops
    template <std::size_t I, std::size_t N, typename Func>
    inline void static_loop(
        const std::array<int, N>& dims,
        std::array<int, N>& idx,
        Func&& f
    ) noexcept
    {
        if constexpr (I == N) {
            f(idx);
        } else {
            for (idx[I] = 0; idx[I] < dims[I]; ++idx[I]) {
                static_loop<I + 1>(dims, idx, f);
            }
        }
    }

    // Core contraction
    template <std::size_t N, typename FA, typename FB>
    inline double contract(
        const std::array<int, N>& dims,
        FA&& A,
        FB&& B
    )
    {
//#ifndef NDEBUG
        for (std::size_t i = 0; i < N; ++i) {
            if (dims[i] <= 0)
                throw std::runtime_error("contract(): invalid dimension");
        }
//#endif

        double sum = 0.0;
        std::array<int, N> idx;

        static_loop<0>(dims, idx, [&](const auto& i) {
            sum += A(i) * B(i);
        });

        return sum;
    }
}