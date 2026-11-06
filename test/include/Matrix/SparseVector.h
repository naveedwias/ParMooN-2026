#ifndef __SPARSEVECTOR_H__
#define __SPARSEVECTOR_H__

#include <vector>

/// @brief Simple sparse vector class with parallelized dot product.
class SparseVector
{
  public:
    /// @brief Construct an empty SparseVector.
    SparseVector(int length = 0);

    /// @brief Construct a SparseVector with a given length and sparsity
    ///        pattern and initialize it with all zeros.
    ///        Column indices must be increasing.
    SparseVector(int length, const std::vector<int>& columns);

    /// @brief Construct a SparseVector with a given length, sparsity pattern,
    ///        and values.
    ///        Column indices must be increasing.
    SparseVector(int length, const std::vector<int>& columns,
      const std::vector<double>& values, bool trim = false);

    /// @brief Get the nominal length of the vector.
    int get_length() const
    {
      return length;
    }

    /// @brief Number of actual entries in sparsity pattern
    int get_n_entries() const
    {
      return (int)columns.size();
    }

    /// @brief Retrieve sparsity pattern
    const std::vector<int>& get_columns() const
    {
      return columns;
    }

    /// @brief Retrieve values at legal column indices
    const std::vector<double>& get_values() const
    {
      return values;
    }

    /// @brief Get index of a given column in the value array, or -1 if illegal
    int get_index(int i) const;

    /// @brief Get index of a given column in the value array,
    /// or -1 if illegal. Optionally adds column.
    int get_index(int i, bool create);

    /// @brief Is this column legal?
    bool has_index(int i) const
    {
      return get_index(i) >= 0;
    }

    /// @brief Get the value at column i, 0 if not in the sparsity pattern
    double operator[](int i) const;

    /// @brief Get a reference to the value at column i.
    ///        Creates the entry if previously unavailable.
    double& operator[](int i);

    /// @brief Compute the dot product with another sparse vector, optionally
    ///        permitting different lengths.
    double dot(const SparseVector& v, bool permissive = false) const;

  protected:
    int length;

    std::vector<int> columns;
    std::vector<double> values;
};

/// @brief Compute the dot product of two sparse vectors.
double dot(const SparseVector& u, const SparseVector& v);

#endif