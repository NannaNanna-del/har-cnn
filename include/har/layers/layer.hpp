#include "har/tensor.hpp"
#include <concepts>

namespace har::layers {
template <std::floating_point T = float> class Layer {
public:
  virtual ~Layer() = default;

  virtual auto forward(const Tensor<T> &input) -> Tensor<T> = 0;
  virtual auto backward(const Tensor<T> &grad_output) -> Tensor<T> = 0;

  virtual auto name() const -> std::string = 0;
  auto output() const -> const Tensor<T> & { return output_; }

protected:
  Tensor<T> input_cache_;
  Tensor<T> output_;
};
} // namespace har::layers
