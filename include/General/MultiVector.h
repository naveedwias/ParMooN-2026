#ifndef __MULTIVECTOR_H__
#define __MULTIVECTOR_H__

#include <array>
#include <vector>
#include <memory>

template<typename T>
class Slice
{
  public:
    Slice(const T* storage, size_t stride, size_t length);
    Slice(T* storage, size_t stride, size_t length, bool constant);

    Slice<T> truncate(size_t n) const;

    const T& at(size_t i) const;
    const T& operator[](size_t i) const;
    T& at(size_t i);
    T& operator[](size_t i);

    size_t size() const;

    double dot(const Slice<T> &other) const;
    double dot() const
    {
      return dot(*this);
    }

    void add(double* v, double a = 1.0) const;
    void product(double* v, const Slice<T> &other, double a = 1.0) const;
    void tensor_product(double** M, const Slice<T> &other, double a = 1.0) const;
    void product(double* v, double a = 1.0) const
    {
      product(v, *this, a);
    }
    void tensor_product(double** M, double a = 1.0) const
    {
      tensor_product(M, *this, a);
    }

  protected:
    const T* storage;
    T* mut_storage;

    const size_t stride;
    const size_t length;
};

/// @brief Multidimensional vector class with automated index management etc.
template<typename T, size_t d>
class MultiVector
{
  public:
    MultiVector();
    MultiVector(const std::array<size_t, d> &dimensions);
    MultiVector(std::vector<T>& storage, const std::array<size_t, d> &dimensions);

    ~MultiVector() = default;

    size_t size();
    void resize(const std::array<size_t, d> &dimensions);
    void fill(const T& value);
    T* data();
    const T* data() const;

    size_t flat_index(const std::array<size_t, d> &index) const;
    size_t flat_index_check(const std::array<size_t, d> &index) const;

    const T& at(const std::array<size_t, d> &index) const;
    T& at(const std::array<size_t, d> &index);

    const T& operator[](const std::array<size_t, d> &index) const;
    T& operator[](const std::array<size_t, d> &index);

    MultiVector<T, d>& operator=(MultiVector<T, d>&) = default;

    const Slice<T> slice(size_t axis, const std::array<size_t, d> index) const;
    const Slice<T> slice(size_t axis, const std::array<size_t, d> index, size_t n) const;
    Slice<T> slice(size_t axis, const std::array<size_t, d> index, bool constant);

    double slice_sum(size_t axis, const std::array<size_t, d> index) const;

  protected:
    std::array<size_t, d> dimensions;
    std::array<size_t, d> strides;
    std::shared_ptr<std::vector<T>> storage;
};

#endif