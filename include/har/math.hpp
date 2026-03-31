#include "har/tensor.hpp"
#include <stdexcept>
namespace har::math {
template <std::floating_point T>

// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N] -> C: [M, N]
auto matmul(const Tensor<T> &a, const Tensor<T> &b) -> Tensor<T> {
  if (a.dims() != 2 || b.dims() != 2) {
    throw std::invalid_argument("matmul requires 2D tensors");
  }

  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  if (b.shape()[0] != K) {
    throw std::invalid_argument("matmul dimension mismatch");
  }

  Tensor<T> c({M, N}, T{0});

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      T a_ik = a.at(i, k);
      for (size_t j = 0; j < N; ++j) {
        c.at(i, j) += a_ik * b.at(k, j);
      }
    }
  }

  return c;
}

// Vector dot product
template <std::floating_point T>
auto dot(const Tensor<T> &a, const Tensor<T> &b) -> T {
  if (a.dims() != 1 || b.dims() != 1 || a.size() != b.size()) {
    throw std::invalid_argument("dot requires 1D tensors of same size");
  }

  T result{0};

  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }

  return result;
}

// Outer product: a @ b
// a: [M], b: [N] -> result: [M, N]
template <std::floating_point T>
auto outer(const Tensor<T> &a, const Tensor<T> &b) -> Tensor<T> {
  if (a.dims() != 1 || b.dims() != 1) {
    throw std::invalid_argument("outer requires 1D tensors");
  }

  const size_t M = a.size();
  const size_t N = b.size();
  Tensor<T> result({M, N});

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result.at(i, j) = a[i] * b[j];
    }
  }

  return result;
}

} // namespace har::math
