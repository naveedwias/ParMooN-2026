#include <MooNMD_Io.h>
#include <SparseVector.h>
#include <Utilities.h>

SparseVector::SparseVector(int length)
: SparseVector(length, {})
{
}

SparseVector::SparseVector(int length, const std::vector<int>& columns)
: length(length), columns(columns)
{
  int c_prev = -1;

  for (int c: columns)
  {
    if (c < 0 || c >= length)
    {
      ErrThrow("Sparse vector of length ", length,
        " includes invalid column index ", c);
    }

    if (c <= c_prev)
    {
      ErrThrow("Sparse vector includes nonincreasing columns: ", c_prev, " -> ", c);
    }

    c_prev = c;
  }

  values.resize(columns.size(), 0.0);
}

SparseVector::SparseVector(int length, const std::vector<int>& columns,
  const std::vector<double>& values, bool trim)
: SparseVector(length, columns)
{
  if (values.size() != columns.size())
  {
    ErrThrow("Sparse vector has ", values.size(), " values for ",
      columns.size(), " columns");
  }

  if (trim)
  {
    unsigned int n = columns.size();

    this->columns.clear();
    this->values.clear();

    for (unsigned int i = 0; i < n; i++)
    {
      if (values[i] != 0.0)
      {
        this->columns.push_back(columns[i]);
        this->values.push_back(values[i]);
      }
    }
  }
  else
  {
    this->values = values;
  }
}

int SparseVector::get_index(int i) const
{
  if (i < 0 || i >= length)
  {
    ErrThrow("Tried to access entry ", i, " of sparse vector of length ", length);
  }

  return utilities::binarySearch(columns, i);
}

int SparseVector::get_index(int i, bool create)
{
  if (i < 0 || i >= length)
  {
    ErrThrow("Tried to access entry ", i, " of sparse vector of length ", length);
  }

  // Find i or the position at which i should be inserted using a linear search

  unsigned int j = 0;

  for (; j < columns.size(); j++)
  {
    int jj = columns[j];

    if (jj == i)
    {
      return j;
    }

    if (jj > i)
    {
      break;
    }
  }

  if (create)
  {
    // Insert a new column
    columns.insert(columns.begin() + j, i);
    values.insert(values.end() + j, 0.0);
    return j;
  }

  return -1;
}

double SparseVector::operator[](int i) const
{
  int k = get_index(i);
  return k < 0 ? 0.0 : values[k];
}

double& SparseVector::operator[](int i)
{
  int k = get_index(i, true);
  return values[k];
}

double SparseVector::dot(const SparseVector& v, bool permissive) const
{
  if (!permissive && v.length != length)
  {
    ErrThrow("Tried to compute dot product of sparse vectors of different lengths");
  }

  int n0 = (int)columns.size();
  int n1 = (int)v.columns.size();

  if (n0 == 0 || n1 == 0
    || columns[n0 - 1] < v.columns[0]
    || v.columns[n1 - 1] < columns[0])
  {
    // no overlap
    return 0.0;
  }
  else
  {
    int i0 = 0;
    int i1 = 0;

    double p = 0.0;

    // loop through both vectors and accumulate the product wherever they meet
    while (i0 < n0 && i1 < n1)
    {
      const int& c0 = columns[i0];
      const int& c1 = v.columns[i1];

      if (c0 == c1)
      {
        p += values[i0++] * v.values[i1++];
      }
      else if (c0 < c1)
      {
        ++i0;
      }
      else
      {
        ++i1;
      }
    }

    return p;
  }
}

double dot(const SparseVector& u, const SparseVector& v)
{
  return u.dot(v);
}