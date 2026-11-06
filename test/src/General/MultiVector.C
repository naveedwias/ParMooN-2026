#include <MultiVector.h>
#include <MooNMD_Io.h>

template<typename T, size_t d>
MultiVector<T, d>::MultiVector()
{
  storage = std::make_shared<std::vector<T>>(0);
  for (size_t i = 0; i < d; i++)
  {
    dimensions[i] = 0;
  }
}

template<typename T, size_t d>
MultiVector<T, d>::MultiVector(const std::array<size_t, d> &dimensions)
{
  storage = std::make_shared<std::vector<T>>(0);
  resize(dimensions);
}

template<typename T, size_t d>
MultiVector<T, d>::MultiVector(std::vector<T>& storage, const std::array<size_t, d> &dimensions)
: dimensions(dimensions)
{
  this->storage.reset(&storage);

  resize(dimensions);
}

template<typename T, size_t d>
size_t MultiVector<T, d>::size()
{
  return storage->size();
}

template<typename T, size_t d>
void MultiVector<T, d>::resize(const std::array<size_t, d> &dimensions)
{
  this->dimensions = dimensions;

  size_t total_size = 1;
  for (int i = (int)d - 1; i >= 0; i--)
  {
    strides[i] = total_size;
    total_size *= dimensions[i];
  }

  storage->resize(total_size);
}

template<typename T, size_t d>
void MultiVector<T, d>::fill(const T& value)
{
  size_t n = storage->size();
  for (size_t i = 0; i < n; i++)
  {
    storage->operator[](i) = value;
  }
}

template<typename T, size_t d>
T* MultiVector<T, d>::data()
{
  return storage->data();
}

template<typename T, size_t d>
const T* MultiVector<T, d>::data() const
{
  return storage->data();
}

template<typename T, size_t d>
size_t MultiVector<T, d>::flat_index(const std::array<size_t, d> &index) const
{
  size_t ind = 0;
  for (size_t i = 0; i < d; i++)
  {
    ind += strides[i] * index[i];
  }

  return ind;
}

template<typename T, size_t d>
size_t MultiVector<T, d>::flat_index_check(const std::array<size_t, d> &index) const
{
  size_t ind = 0;
  for (size_t i = 0; i < d; i++)
  {
    if (index[i] >= dimensions[i])
    {
      std::ostringstream os;

      os << "Index out of range: (";
      for (size_t i = 0; i < d; i++)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << index[i];
      }
      os << "). Dimensions are (";
      for (size_t i = 0; i < d; i++)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << dimensions[i];
      }
      os << ").";

      ErrThrow(os.str());
    }

    ind += strides[i] * index[i];
  }

  return ind;
}

template<typename T, size_t d>
const T& MultiVector<T, d>::at(const std::array<size_t, d> &index) const
{
  return storage->at(flat_index_check(index));
}

template<typename T, size_t d>
T& MultiVector<T, d>::at(const std::array<size_t, d> &index)
{
  return storage->at(flat_index_check(index));
}

template<typename T, size_t d>
const T& MultiVector<T, d>::operator[](const std::array<size_t, d> &index) const
{
  return storage->operator[](flat_index(index));
}

template<typename T, size_t d>
T& MultiVector<T, d>::operator[](const std::array<size_t, d> &index)
{
  return storage->operator[](flat_index(index));
}

template<typename T, size_t d>
const Slice<T> MultiVector<T, d>::slice(size_t axis, const std::array<size_t, d> index) const
{
  if (axis >= d)
  {
    ErrThrow("Tried to slice a ", d, "-vector on axis ", axis);
  }

  size_t start = flat_index_check(index);
  size_t stride = strides[axis];
  size_t length = dimensions[axis] - index[axis];

  return Slice<T>(storage->data() + start, stride, length, true);
}

template<typename T, size_t d>
const Slice<T> MultiVector<T, d>::slice(size_t axis, const std::array<size_t, d> index, size_t n) const
{
  if (axis >= d)
  {
    ErrThrow("Tried to slice a ", d, "-vector on axis ", axis);
  }

  size_t start = flat_index_check(index);
  size_t stride = strides[axis];
  size_t length = dimensions[axis] - index[axis];

  if (length < n)
  {
    ErrThrow("Tried to extend a slice of length ", length, " to length ", n,
      "! Only truncation is allowed.")
  }

  return Slice<T>(storage->data() + start, stride, n, true);
}

template<typename T, size_t d>
Slice<T> MultiVector<T, d>::slice(size_t axis, const std::array<size_t, d> index, bool constant)
{
  if (axis >= d)
  {
    ErrThrow("Tried to slice a ", d, "-vector on axis ", axis);
  }

  size_t start = flat_index_check(index);
  size_t stride = strides[axis];
  size_t length = dimensions[axis] - index[axis];

  return Slice<T>(storage->data() + start, stride, length, constant);
}

template<typename T, size_t d>
double MultiVector<T, d>::slice_sum(size_t axis, const std::array<size_t, d> index) const
{
  if (axis >= d)
  {
    ErrThrow("Tried to slice a ", d, "-vector on axis ", axis);
  }

  size_t flat = flat_index_check(index);
  size_t stride = strides[axis];
  size_t length = dimensions[axis] - index[axis];

  double sum = 0.0;

  const double* data = storage->data();

  for (size_t i = 0; i < length; i++)
  {
    sum += data[flat];
    flat += stride;
  }

  return sum;
}

template<typename T>
Slice<T>::Slice(const T* storage, size_t stride, size_t length)
: storage(storage), mut_storage(nullptr), stride(stride), length(length)
{
}

template<typename T>
Slice<T>::Slice(T* storage, size_t stride, size_t length, bool constant)
: storage(storage), mut_storage(constant ? nullptr : storage), stride(stride), length(length)
{
}

template<typename T>
Slice<T> Slice<T>::truncate(size_t n) const
{
  if (length < n)
  {
    ErrThrow("Can't truncate length ", length, " to ", n, "!");
  }

  if (mut_storage != nullptr)
  {
    return Slice<T>(mut_storage, stride, n, false);
  }
  else
  {
    return Slice<T>(storage, stride, n);
  }
}

template<typename T>
const T& Slice<T>::at(size_t i) const
{
  if (i >= length)
  {
    ErrThrow("Tried to access entry ", i, " of a slice of length ", length);
  }

  return storage[stride * i];
}

template<typename T>
const T& Slice<T>::operator[](size_t i) const
{
  return storage[stride * i];
}

template<typename T>
T& Slice<T>::at(size_t i)
{
  if (!mut_storage)
  {
    ErrThrow("Tried to retrieve non-constant entry from constant slice!");
  }

  if (i >= length)
  {
    ErrThrow("Tried to access entry ", i, " of a slice of length ", length);
  }

  return mut_storage[stride * i];
}

template<typename T>
T& Slice<T>::operator[](size_t i)
{
  return mut_storage[stride * i];
}

template<typename T>
size_t Slice<T>::size() const
{
  return length;
}

template<typename T>
double Slice<T>::dot(const Slice<T> &other) const
{
  if (other.length != length)
  {
    ErrThrow("Mismatched slice length for dot product: ",
      length, " vs. ", other.length);
  }

  double prod = 0.0;

  for (unsigned int i = 0; i < length; i++)
  {
    prod += storage[i * stride] * other.storage[i * other.stride];
  }

  return prod;
}

template<typename T>
void Slice<T>::add(double* v, double a) const
{
  for (unsigned int i = 0; i < length; i++)
  {
    v[i] += a * storage[i * stride];
  }
}

template<typename T>
void Slice<T>::product(double* v, const Slice<T> &other, double a) const
{
  if (other.length != length)
  {
    ErrThrow("Mismatched slice length for product: ",
      length, " vs. ", other.length);
  }

  for (unsigned int i = 0; i < length; i++)
  {
    v[i] += a * storage[i * stride] * other.storage[i * other.stride];
  }
}

template<typename T>
void Slice<T>::tensor_product(double** M, const Slice<T> &other, double a) const
{
  for (unsigned int i = 0; i < length; i++)
  {
    double* c = M[i];
    const double m = a * storage[i * stride];

    for (unsigned int j = 0; j < other.length; j++)
    {
      c[j] += m * other.storage[j * other.stride];
    }
  }
}

template class Slice<double>;
template class MultiVector<double, 2>;
template class MultiVector<double, 3>;