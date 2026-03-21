#pragma once

#include <array>
#include <concepts>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace har {
template <std::floating_point T = float> class Tensor {
private:
  std::vector<size_t> shape_;
  std::vector<T> data_;

  auto compute_size() const -> size_t {
    if (shape_.empty()) {
      return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                           std::multiplies{});
  }

  template <typename... Indices>
  auto compute_flat_index(Indices... indices) const -> size_t {
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    if (idx_arr.size() != shape_.size()) {
      throw std::invalid_argument("Index dimension mismatch");
    }

    size_t flat_idx = 0;
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
      flat_idx += idx_arr[i] * stride;
      stride *= shape_[i];
    }

    return flat_idx;
  }

public:
  Tensor() = default;

  explicit Tensor(std::vector<size_t> shape)
      : shape_(std::move(shape)), data_(compute_size()) {}

  Tensor(std::vector<size_t> shape, T init_value)
      : shape_(std::move(shape)), data_(compute_size(), init_value) {}

  auto shape() const -> const std::vector<size_t> & { return shape_; }

  auto size() const -> size_t { return data_.size(); }

  auto data() -> T * { return data_.data(); }

  auto data() const -> const T * { return data_.data(); }

  auto begin() { return data_.begin(); }

  auto end() { return data_.end(); }

  auto begin() const { return data_.begin(); }

  auto end() const { return data_.end(); }

  auto operator[](size_t idx) -> T & { return data_[idx]; }

  auto operator[](size_t idx) const -> const T & { return data_[idx]; }

  auto dims() const -> size_t { return shape_.size(); }

  template <typename... Indices>
    requires(sizeof...(Indices) > 0) &&
            (std::convertible_to<Indices, size_t> && ...)
  auto at(Indices... indices) -> T & {
    return data_[compute_flat_index(indices...)];
  }

  template <typename... Indices>
    requires(sizeof...(Indices) > 0) &&
            (std::convertible_to<Indices, size_t> && ...)
  auto at(Indices... indices) const -> const T & {
    return data_[compute_flat_index(indices...)];
  }

  void reshape(std::vector<size_t> new_shape) {
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                    size_t{1}, std::multiplies{});

    if (new_size != data_.size()) {
      throw std::invalid_argument("Reshape size mismatch");
    }

    shape_ = std::move(new_shape);
  }

  void fill(T value) { std::ranges::fill(data_, value); }
};

} // namespace har
