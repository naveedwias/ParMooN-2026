#include <Utilities.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <MooNMD_Io.h>

std::string utilities::get_host_name()
{
  char buf[80];
  gethostname(buf,80);
  return std::string(buf);
}


std::string utilities::get_date_and_time()
{
  auto time = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(time);
  return std::ctime(&t);
}

double utilities::minmod(const std::vector< double >& input_vec)
{
  if (input_vec.empty())
  {
    ErrThrow("The input vector input_vec is empty. Verify that it is filled ",
        "with at least one entry.");
  }
  else
  {
    double return_val = (input_vec[0] != 0) ? input_vec[0] : 0;
    if (input_vec[0] != 0 && input_vec.size() > 1)
    {
      // if one entry is zero minmod has to return 0 directly.
      double sign = input_vec[0] / std::abs(input_vec[0]);
      for (auto elem : input_vec)
      {
        if (elem == input_vec[0])
        {
          // the first entry or entries that have been checked before do not
          // have to be tested again
          continue;
        }
        if (elem == 0)
        {
          // if one entry is zero minmod has to return 0 directly.
          return_val = 0;
          break;
        }
        else if ( (sign > 0 && elem < 0) || (sign < 0 && elem > 0) )
        {
          // signs differ
          return_val = 0;
          break;
        }
        else if (std::abs(elem) < sign * return_val)
        {
          // new "minimal" value
          return_val = elem;
        }
      }
    }
    return return_val;
  }
}

bool utilities::are_equal(double a, double b, double tol)
{
  bool are_equal = (std::abs(a) < 1) ? std::abs(a-b) < tol :
    std::abs(a-b) < tol * std::abs(a);
  return are_equal;
}

void utilities::copy_vec(std::vector<double> &v, const std::vector<double> &w)
{
  unsigned int n = v.size();

  if (w.size() != n)
  {
    ErrThrow("Vector size mismatch!");
  }

  memcpy(v.data(), w.data(), n * sizeof(double));
}

void utilities::clear_vec(std::vector<double> &v)
{
  memset(v.data(), 0, v.size() * sizeof(double));
}

void utilities::clear_vec(std::vector<double> &v, double z)
{
  unsigned int n = v.size();
  for (unsigned int i = 0; i < n; i++)
  {
    v[i] = z;
  }
}

void utilities::scale_vec(std::vector<double> &v, double a)
{
  unsigned int n = v.size();
  for (unsigned int i = 0; i < n; i++)
  {
    v[i] *= a;
  }
}

void utilities::add_vec(std::vector<double> &v, const std::vector<double> &w, double a)
{
  unsigned int n = v.size();

  if (w.size() != n)
  {
    ErrThrow("Vector size mismatch!");
  }

  for (unsigned int i = 0; i < n; i++)
  {
    v[i] += a * w[i];
  }
}

double utilities::dot_vec(const std::vector<double> &v, const std::vector<double> &w)
{
  unsigned int n = v.size();

  if (w.size() != n)
  {
    ErrThrow("Vector size mismatch!");
  }

  double d = 0.0;

  for (unsigned int i = 0; i < n; i++)
  {
    d += v[i] * w[i];
  }

  return d;
}

double utilities::norm_vec(const std::vector<double> &v)
{
  unsigned int n = v.size();

  double d = 0.0;

  for (unsigned int i = 0; i < n; i++)
  {
    d += v[i] * v[i];
  }

  return std::sqrt(d);
}

double utilities::normi_vec(const std::vector<double> &v)
{
  unsigned int n = v.size();

  double d = 0.0;

  for (unsigned int i = 0; i < n; i++)
  {
    d = std::max(d, std::abs(v[i]));
  }

  return d;
}

void utilities::copy_vec(double* v, const double* w, int n)
{
  memcpy(v, w, n * sizeof(double));
}

void utilities::copy_vec(double* v, const double* w0, const double* w1, int n, int crossover)
{
  memcpy(v, w0, crossover * sizeof(double));
  memcpy(v + crossover, w1 + crossover, (n - crossover) * sizeof(double));
}

void utilities::clear_vec(double* v, int n)
{
  memset(v, 0, n * sizeof(double));
}

void utilities::clear_vec(double* v, int n, double z)
{
  for (int i = 0; i < n; i++)
  {
    v[i] = z;
  }
}

void utilities::scale_vec(double* v, int n, double a)
{
  for (int i = 0; i < n; i++)
  {
    v[i] *= a;
  }
}

double utilities::dot_vec(const double* v, const double* w, int n)
{
  double d = 0.0;

  for (int i = 0; i < n; i++)
  {
    d += v[i] * w[i];
  }

  return d;
}

double utilities::norm_vec(const double* v, int n)
{
  double d = 0.0;

  for (int i = 0; i < n; i++)
  {
    d += v[i] * v[i];
  }

  return std::sqrt(d);
}

double utilities::normi_vec(const double* v, int n)
{
  double d = 0.0;

  for (int i = 0; i < n; i++)
  {
    d = std::max(d, std::abs(v[i]));
  }

  return d;
}

void utilities::add_vec(double* v, const double* w, int n, double a)
{
  for (int i = 0; i < n; i++)
  {
    v[i] += a * w[i];
  }
}

void utilities::add_vec(double* v, const double* w0, const double* w1, int n, int crossover, double a)
{
  for (int i = 0; i < crossover; i++)
  {
    v[i] += a * w0[i];
  }

  for (int i = crossover; i < n; i++)
  {
    v[i] += a * w1[i];
  }
}

void utilities::add_mat(double** A, const double* const* B, int n, int m, double a)
{
  if (m < 0)
  {
    m = n;
  }

  for (int i = 0; i < n; i++)
  {
    double* ai = A[i];
    const double* bi = B[i];

    for (int j = 0; j < m; j++)
    {
      ai[j] += a * bi[j];
    }
  }
}

double utilities::norm_mat(const double* const* A, int n, int m)
{
  if (m < 0)
  {
    m = n;
  }

  double f = 0.0;

  for (int i = 0; i < n; i++)
  {
    const double* ai = A[i];

    for (int j = 0; j < m; j++)
    {
      f += ai[j] * ai[j];
    }
  }

  return std::sqrt(f);
}

template<typename T>
int utilities::binarySearch(const T* arr, size_t n, const T& value)
{
  if (n < 1)
  {
    return -1;
  }

  size_t a = 0;
  size_t b = n - 1;
  T va = arr[a];
  T vb = arr[b];

  if (va == value)
  {
    return a;
  }
  else if (vb == value)
  {
    return b;
  }
  else if (va > value)
  {
    return -1;
  }
  else if (vb < value)
  {
    return -1;
  }
  else
  {
    while (true)
    {
      size_t c = (a + b) / 2;
      T vc = arr[c];

      if (c == a || c == b)
      {
        return -1;
      }

      if (vc == value)
      {
        return c;
      }
      else if (vc > value)
      {
        b = c;
        vb = vc;
      }
      else
      {
        a = c;
        va = vc;
      }
    }

    return -1;
  }
}

template int utilities::binarySearch<int>(const int*, size_t, const int&);
template int utilities::binarySearch<unsigned int>(const unsigned int*, size_t, const unsigned int&);
template int utilities::binarySearch<size_t>(const size_t*, size_t, const size_t&);
template int utilities::binarySearch<double>(const double*, size_t, const double&);