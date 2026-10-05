#pragma once

#include <cstddef>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <stdexcept>

// utility functions for open shell and closed shell Optimization and Integrals classes

using namespace madness;
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

namespace refinement_utils {
    struct NumericalParameters {
        double truncation_tol = 1e-6;
        double coulomb_lo = 0.001;
        double coulomb_eps = 1e-6;
        double BSH_lo = 0.001;
        double BSH_eps = 1e-6;
    };

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

    inline void TransformMatrix(
        madness::Tensor<double>* ObjectMatrix,
        madness::Tensor<double>& TransformationMatrix)
    {
        madness::Tensor<double> temp = inner(*ObjectMatrix, TransformationMatrix);
        *ObjectMatrix = inner(transpose(TransformationMatrix), temp);
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

    template <std::size_t NDIM>
    inline std::vector<Function<double, NDIM>> orthonormalize_mixed_by_degeneracy(
        madness::World& world,
        std::vector<Function<double, NDIM>>& orbitals,
        const std::vector<double>& occupations,
        double degeneracy_tol)
    {
        std::cout << "\n=== Mixed Orthonormalization ===" << std::endl;

        const int n_orb = static_cast<int>(occupations.size());
        for (int i = 0; i < n_orb; ++i) {
            std::cout << "Orbital " << i << " occupation: " << occupations[i] << std::endl;
        }

        std::vector<std::pair<int, int>> groups;
        int i = 0;
        while (i < n_orb) {
            int start = i;
            double current_occ = occupations[i];

            int j = i + 1;
            while (j < n_orb && std::abs(occupations[j] - current_occ) < degeneracy_tol) {
                ++j;
            }

            groups.emplace_back(start, j);
            i = j;
        }

        std::cout << "Found " << groups.size() << " degeneracy groups:" << std::endl;

        std::vector<Function<double, NDIM>> result_orbitals;
        for (size_t g = 0; g < groups.size(); ++g) {
            int start = groups[g].first;
            int end = groups[g].second;
            int group_size = end - start;

            std::vector<Function<double, NDIM>> group_orbitals;
            for (int k = start; k < end; ++k) {
                group_orbitals.push_back(orbitals[k]);
            }

            std::vector<Function<double, NDIM>> ortho_group_orbitals;
            if (group_size == 1) {
                std::cout << "  Group " << g << " (orbital " << start << "): "
                          << "occupation=" << occupations[start] << ", method=Cholesky" << std::endl;

                if (result_orbitals.size() > 0) {
                    auto current_orb = group_orbitals[0];
                    for (const auto& prev_orb : result_orbitals) {
                        double overlap = madness::inner(current_orb, prev_orb);
                        current_orb = current_orb - overlap * prev_orb;
                    }

                    double norm = current_orb.norm2();
                    if (norm > 1e-12) {
                        current_orb.scale(1.0 / norm);
                    }
                    ortho_group_orbitals.push_back(current_orb);
                } else {
                    double norm = group_orbitals[0].norm2();
                    group_orbitals[0].scale(1.0 / norm);
                    ortho_group_orbitals = group_orbitals;
                }
            } else {
                std::cout << "  Group " << g << " (orbitals " << start << "-" << (end - 1) << "): "
                          << "occupations=[";
                for (int k = start; k < end; ++k) {
                    std::cout << occupations[k];
                    if (k < end - 1) std::cout << ", ";
                }
                std::cout << "], method=Symmetric (within group)" << std::endl;

                if (result_orbitals.size() > 0) {
                    for (auto& group_orb : group_orbitals) {
                        for (const auto& prev_orb : result_orbitals) {
                            double overlap = madness::inner(group_orb, prev_orb);
                            group_orb = group_orb - overlap * prev_orb;
                        }
                    }
                }

                auto S = madness::matrix_inner(world, group_orbitals, group_orbitals, true);
                ortho_group_orbitals = madness::orthonormalize_symmetric(group_orbitals, S);
            }

            for (auto& orb : ortho_group_orbitals) {
                result_orbitals.push_back(orb);
            }
        }

        std::cout << "=== Mixed Orthonormalization Complete ===\n" << std::endl;
        return result_orbitals;
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