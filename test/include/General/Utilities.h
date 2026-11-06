#ifndef PARMOON__UTILITIES
#define PARMOON__UTILITIES

#include <string>
#include <vector>

namespace utilities
{
  /// @brief return the host name of the computer ParMooN is running on
  std::string get_host_name();

  /// @brief return the date and time as a string
  std::string get_date_and_time();

  /** @brief returns minmod of a vector of numbers
   *
   * @details This function is the implementation of \f$\mathrm{minmod}(a_0, a_1,
   * \ldots, a_{N-1})\f$, \f$N\ge 1\f$, for \f$N\f$ input values where
   * \f$\mathrm{minmod}\f$ is defined by
   * \f[
   *     \mathrm{minmod}(a_0, a_1, \ldots, a_{N-1}):=
   *       \begin{cases}
   *          s \min_{i = 1, 2, \ldots, N-1}{\vert a_i \vert} &\text{ if } s :=
   *          \mathrm{sign}(a_0) = \mathrm{sign}(a_1) = \ldots =
   *          \mathrm{sign}(a_{N-1}) \\
   *          0 & \text{ else.}
   *        \end{cases}
   * \f]
   * @param[in] input_vec vector of numbers \f$a_0, a_1,
   * ..., a_{N-1}\f$
   * @return \f$\mathrm{minmod}(a_0, a_1,\ldots, a_{N-1})\f$
   */
  double minmod(const std::vector< double >& input_vec);

  /**
   * @brief checks if two numbers are equal up to given tolerance
   *
   * @details If the first number \f$|a|\ge 1\f$, the relative error
   *\f[
       |a-b| < \mathrm{tol} \cdot |a|
   \f]
   * between \p a and \p b is investigated. If \f$|a| < 1\f$ the absolute error
   *\f[
     |a-b| < tol
   \f]
   * is checked.
   *
   * The default value for \p tol is 1e-12.
   */
  bool are_equal(double a, double b, double tol = 1e-12);

  void copy_vec(std::vector<double> &v, const std::vector<double> &w);

  void clear_vec(std::vector<double> &v);

  void clear_vec(std::vector<double> &v, double z);

  void scale_vec(std::vector<double> &v, double a);

  double dot_vec(const std::vector<double> &v, const std::vector<double> &w);

  double norm_vec(const std::vector<double> &v);

  double normi_vec(const std::vector<double> &v);

  void add_vec(std::vector<double> &v, const std::vector<double> &w, double a = 1.0);

  void copy_vec(double* v, const double* w, int n);

  void copy_vec(double* v, const double* w0, const double* w1, int n, int crossover);

  void clear_vec(double* v, int n);

  void clear_vec(double* v, int n, double z);

  void scale_vec(double* v, int n, double a);

  double dot_vec(const double* v, const double* w, int n);

  double norm_vec(const double* v, int n);

  double normi_vec(const double* v, int n);

  void add_vec(double* v, const double* w, int n, double a = 1.0);

  void add_vec(double* v, const double* w0, const double* w1, int n, int crossover, double a = 1.0);

  void add_mat(double** A, const double* const* B, int n, int m = -1, double a = 1.0);

  double norm_mat(const double* const* A, int n, int m = -1);

  /// @brief Binary search in a sorted array
  template<typename T> int binarySearch(const T* arr, size_t n, const T& value);

  /// @brief Binary search in a sorted vector
  template<typename T> int binarySearch(const std::vector<T> &arr, const T& value)
  {
    if (arr.size() == 0)
    {
      return -1;
    }

    return binarySearch(arr.data(), arr.size(), value);
  }

  /// @brief Binary search in a sorted vector
  template<typename T> int binarySearch(const std::vector<T> &arr, int a, int b, const T& value)
  {
    if (a < b && (a < 0 || a >= (int)arr.size() || b < 0 || b > (int)arr.size()))
    {
      ErrThrow("Invalid search range ", a, " ... ", b, " in vector of size ", arr.size());
    }

    return binarySearch(arr.data() + a, b - a, value);
  }
}

#endif// PARMOON__UTILITIES
