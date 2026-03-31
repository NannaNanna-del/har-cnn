#include "har/tensor.hpp"
#include <concepts>
#include <memory>
#include <vector>

template <std::floating_point T = float> struct Parameter {
  har::Tensor<T> data;
  har::Tensor<T> grad;
  bool requires_grad{true};

  Parameter() = default;
  explicit Parameter(std::vector<size_t> shape)
      : data(std::move(shape), grad(std::move(data.shape(), T{0}))) {}
  Parameter(std::vector<size_t> shape, T init_value)
      : data(std::move(shape), init_value), grad(data.shape(), T{0}) {}

  void zero_grad() { grad.zero(); }
};

namespace har::layers {
template <std::floating_point T = float> class Layer {
public:
  virtual ~Layer() = default;

  virtual auto forward(const Tensor<T> &input) -> Tensor<T> = 0;

  virtual auto backward(const Tensor<T> &grad_output) -> Tensor<T> = 0;

  virtual auto name() const -> std::string = 0;

  auto output() const -> const Tensor<T> & { return output_; }

  // Get parameters for optimization
  virtual auto parameters() -> std::vector<Parameter<T> *> { return {}; }

  // Zero all gradients
  virtual void zero_grad() {
    for (auto *param : parameters()) {
      param->zero_grad();
    }
  }

  void set_traning(bool training) { training_ = training; }

  auto is_training() const -> bool { return training_; }

protected:
  Tensor<T> input_cache_;
  Tensor<T> output_;
  bool training_{true};
};

template <std::floating_point T = float>
using LayerPtr = std::unique_ptr<Layer<T>>;
} // namespace har::layers
