#undef TIME_REDUCED_ASSEMBLING

#include <ReducedAssembling.h>
#include <Database.h>
#include <FEDatabase.h>
#include <BaseCell.h>
#include <MooNMD_Io.h>

#ifdef __3D__
#include <FESpace3D.h>
#endif

#ifdef __2D__
#include <FESpace2D.h>
#endif

#include <cmath>
#include <algorithm>
#include <string.h>

#ifdef TIME_REDUCED_ASSEMBLING
#include <Chrono.h>
#endif

template<int d>
ReducedAssembling<d>::ReducedAssembling(
  std::vector<std::shared_ptr<const FESpace>> test_spaces,
  std::vector<std::shared_ptr<const FESpace>> ansatz_spaces,
  std::vector<std::vector<std::shared_ptr<const FESpace>>> aux_spaces,
  int test_dim, int ansatz_dim,
  const double* const* test_basis, const double* const* ansatz_basis,
  const MultiIndex_vector& test_derivatives,
  const MultiIndex_vector& ansatz_derivatives)
: test_spaces(test_spaces), ansatz_spaces(ansatz_spaces),
  aux_spaces(aux_spaces),
  test_vector_dim((int)test_spaces.size()), ansatz_vector_dim((int)ansatz_spaces.size()),
  test_dim(test_dim), ansatz_dim(ansatz_dim),
  test_basis(test_basis), ansatz_basis(ansatz_basis),
  test_derivatives(test_derivatives), ansatz_derivatives(ansatz_derivatives),

  aux_functions(aux_spaces.size()),
  aux_sd(aux_spaces.size()), aux_indices(aux_spaces.size()),
  aux_derivative_lookup(aux_spaces.size()),
  aux_basis_offset(aux_spaces.size()),

  setup_finished(false)
{
  if (test_dim <= 0)
  {
    ErrThrow("Nonpositive test space dimension");
  }

  if (ansatz_dim <= 0)
  {
    ErrThrow("Nonpositive ansatz space dimension");
  }

  if (test_spaces.size() == 0)
  {
    ErrThrow("No test spaces provided")
  }

  if (ansatz_spaces.size() == 0)
  {
    ErrThrow("No ansatz spaces provided")
  }
}

template<int d>
ReducedAssembling<d>::ReducedAssembling(
  std::vector<std::shared_ptr<const FESpace>> spaces,
  std::vector<std::vector<std::shared_ptr<const FESpace>>> aux_spaces,
  int dim, const double* const* basis,
  const MultiIndex_vector& derivatives)
: ReducedAssembling(spaces, spaces,
  aux_spaces,
  dim, dim,
  basis, basis,
  derivatives, derivatives)
{
}

template<int d>
ReducedAssembling<d>::FunctionInfo::FunctionInfo(const double* func,
  const MultiIndex_vector& derivatives,
  const bool is_test)
: func(func), derivatives(derivatives), is_test(is_test)
{
}

template<int d>
void ReducedAssembling<d>::AddReducedFunction(const double* func,
  const MultiIndex_vector& derivatives, bool is_test)
{
  reduced_functions.emplace_back(func, derivatives, is_test || !DistinctBases());

  setup_finished = false;
}

template<int d>
void ReducedAssembling<d>::AddFullFunction(const double* func,
  const MultiIndex_vector& derivatives, bool is_test)
{
  full_functions.emplace_back(func, derivatives, is_test || !DistinctBases());

  setup_finished = false;
}

template<int d>
void ReducedAssembling<d>::AddAuxFunction(const double* func,
      const MultiIndex_vector& derivatives, int aux_space)
{
  if (aux_space < 0 || aux_space > (int)aux_spaces.size())
  {
    ErrThrow("Tried to add aux function for invalid aux space #", aux_space);
  }

  aux_functions[aux_space].emplace_back(func, derivatives, false);

  setup_finished = false;
}

template<int d>
void ReducedAssembling<d>::AddAssembly(AssembleLocalReduced assembly, double m)
{
  assembly_functions.push_back(assembly);
  assembly_multipliers.push_back(m);

  setup_finished = false;
}

template<int d>
void ReducedAssembling<d>::Reset()
{
  reduced_functions.clear();
  full_functions.clear();
  assembly_functions.clear();
  assembly_multipliers.clear();

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    aux_functions[m].clear();
  }

  setup_finished = false;
}

template<int d>
bool ReducedAssembling<d>::DistinctBases()
{
  // check whether test and ansatz spaces or base data are distinct

  int n_test_values = (int)test_derivatives.size();
  int n_ansatz_values = (int)ansatz_derivatives.size();

  if ((test_basis != ansatz_basis)
    || (test_vector_dim != ansatz_vector_dim)
    || (test_dim != ansatz_dim)
    || (n_test_values != n_ansatz_values))
  {
    return true;
  }

  // dimensions etc. are the same - check if everything matches

  for (int i = 0; i < n_test_values; i++)
  {
    if (test_derivatives[i] != ansatz_derivatives[i])
    {
      // derivative order differs, bases are distinct
      return true;
    }
  }

  for (int i = 0; i < test_vector_dim; i++)
  {
    if (test_spaces[i] != ansatz_spaces[i])
    {
      // spaces differ
      return true;
    }
  }
  return false;
}

template<int d>
void ReducedAssembling<d>::SetBasisAndFunctionStorage()
{
  unsigned int n_test_values = test_derivatives.size();
  unsigned int n_ansatz_values = ansatz_derivatives.size();

  // check whether test and ansatz spaces or base data are distinct
  bool distinct_bases = DistinctBases();

  // set up vectors to hold basis values

  test_values.resize({ (unsigned)test_vector_dim, n_test_values, (unsigned)test_dim });

  if (distinct_bases)
  {
    ansatz_values.resize({ (unsigned)ansatz_vector_dim, n_ansatz_values, (unsigned)ansatz_dim });
  }

  // set up vectors to hold reduced FE function values

  reduced_values.resize(reduced_functions.size());
  for (unsigned int i = 0; i < reduced_functions.size(); i++)
  {
    reduced_values[i].resize(
      {
        (unsigned)(full_functions[i].is_test ? test_vector_dim : ansatz_vector_dim),
        reduced_functions[i].derivatives.size()
      });
  }

  // set up vectors to hold full FE function values

  full_values.resize(full_functions.size());
  for (unsigned int i = 0; i < full_functions.size(); i++)
  {
    full_values[i].resize(
      {
        (unsigned)(full_functions[i].is_test ? test_vector_dim : ansatz_vector_dim),
        full_functions[i].derivatives.size()
      });
  }

  // set up vectors to hold aux FE function values

  unsigned int num_aux_functions = 0;
  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    num_aux_functions += aux_functions[m].size();
  }

  aux_values.resize(num_aux_functions);

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    unsigned int vector_dim = aux_spaces[m].size();

    for (unsigned int i = 0; i < aux_functions[m].size(); i++)
    {
      aux_values[i].resize({ vector_dim, aux_functions[m][i].derivatives.size() });
    }
  }
}

template<int d>
void ReducedAssembling<d>::SetDerivativeLookup(const MultiIndex_vector& derivatives,
  const MultiIndex_vector& indices, std::vector<int>& lookup)
{
  lookup.resize(derivatives.size());

  for (unsigned int i = 0; i < derivatives.size(); i++)
  {
    lookup[i] = -1;

    for (unsigned int j = 0; j < indices.size(); j++)
    {
      if (indices[j] == derivatives[i])
      {
        lookup[i] = (int)j;
        break;
      }
    }

    if (lookup[i] < 0)
    {
      ErrThrow("Missing derivative!");
    }
  }
}

template<int d>
void ReducedAssembling<d>::SetDerivativeLookups()
{
  test_sd = needs_second_derivatives<d>(test_derivatives);
  ansatz_sd = needs_second_derivatives<d>(ansatz_derivatives);

  for (const FunctionInfo& f: reduced_functions)
  {
    if (f.is_test)
    {
      test_sd = test_sd || needs_second_derivatives<d>(f.derivatives);
    }
    else
    {
      ansatz_sd = ansatz_sd || needs_second_derivatives<d>(f.derivatives);
    }
  }

  for (const FunctionInfo& f: full_functions)
  {
    if (f.is_test)
    {
      test_sd = test_sd || needs_second_derivatives<d>(f.derivatives);
    }
    else
    {
      ansatz_sd = ansatz_sd || needs_second_derivatives<d>(f.derivatives);
    }
  }

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    aux_sd[m] = false;

    for (const FunctionInfo& f: aux_functions[m])
    {
      if (needs_second_derivatives<d>(f.derivatives))
      {
        aux_sd[m] = true;
        break;
      }
    }
  }

  // create lookup structures to map derivative order to retrieved order

  test_indices = indices_up_to_order<d>(test_sd ? 2 : 1);
  ansatz_indices = indices_up_to_order<d>(ansatz_sd ? 2 : 1);

  SetDerivativeLookup(test_derivatives, test_indices, test_derivative_lookup);
  SetDerivativeLookup(ansatz_derivatives, ansatz_indices, ansatz_derivative_lookup);

  reduced_derivative_lookup.resize(reduced_functions.size());
  full_derivative_lookup.resize(full_functions.size());

  for (unsigned int i = 0; i < reduced_functions.size(); i++)
  {
    const FunctionInfo& f = reduced_functions[i];

    if (f.is_test)
    {
      SetDerivativeLookup(f.derivatives, test_derivatives, reduced_derivative_lookup[i]);
    }
    else
    {
      SetDerivativeLookup(f.derivatives, ansatz_derivatives, reduced_derivative_lookup[i]);
    }
  }

  for (unsigned int i = 0; i < full_functions.size(); i++)
  {
    const FunctionInfo& f = full_functions[i];

    if (f.is_test)
    {
      SetDerivativeLookup(f.derivatives, test_derivatives, full_derivative_lookup[i]);
    }
    else
    {
      SetDerivativeLookup(f.derivatives, ansatz_derivatives, full_derivative_lookup[i]);
    }
  }

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    aux_indices[m] = indices_up_to_order<d>(aux_sd[m] ? 2 : 1);

    aux_derivative_lookup[m].resize(aux_functions[m].size());

    for (unsigned int i = 0; i < aux_functions[m].size(); i++)
    {
      const FunctionInfo& f = aux_functions[m][i];

      SetDerivativeLookup(f.derivatives, aux_indices[m], aux_derivative_lookup[m][i]);
    }
  }
}

template<int d>
void ReducedAssembling<d>::RetrieveLocalElements(const std::vector<std::shared_ptr<const FESpace>>& spaces,
  int c, unsigned int offset)
{
  for (unsigned int i = 0; i < spaces.size(); i++)
  {
    auto& element = spaces[i]->get_fe(c);
    elements[offset + i] = &element;
    bf[offset + i] = element.GetBaseFunct();
  }
}

template<int d>
void ReducedAssembling<d>::RetrieveLocalBasis(const std::vector<std::shared_ptr<const FESpace>>& spaces,
  const MultiIndex_vector& indices, unsigned int offset)
{
  std::shared_ptr<const FESpace> last_space = nullptr;

  for (unsigned int i = 0; i < spaces.size(); i++)
  {
    unsigned int ii = offset + i;
    orig_basis_values[ii].resize(indices.size());

    if (last_space != spaces[i])
    {
      for (unsigned int j = 0; j < indices.size(); j++)
      {
        // no need to pass cell information here because we called
        // FEDatabase::GetOrig() before this
        orig_basis_values[ii][j] = FEDatabase::GetOrigElementValues(
          *bf[ii], indices[j]);
      }
    }
    else
    {
      for (unsigned int j = 0; j < indices.size(); j++)
      {
        orig_basis_values[ii][j] = orig_basis_values[ii - 1][j];
      }
    }

    last_space = spaces[i];
  }
}

template<int d>
TQuadFormula ReducedAssembling<d>::RetrieveLocalFEData(
  const TCollection* collection, int c, bool* sd)
{
  const TBaseCell* cell = collection->GetCell(c);

  // retrieve local FE information

  RetrieveLocalElements(test_spaces, c, 0);
  RetrieveLocalElements(ansatz_spaces, c, test_spaces.size());

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    RetrieveLocalElements(aux_spaces[m], c, aux_basis_offset[m]);
  }

  TQuadFormula qf_ref(QuadratureFormula_type::BaryCenterTetra);
  TQuadFormula qf_orig(qf_ref);

  // retrieve common quadrature formula and set cell
  // (FEDatabase has ugly hidden state)

  FEDatabase::GetOrig(
    elements, collection, cell, sd,
    qf_ref, qf_orig);

  // retrieve local full-order FE bases

  RetrieveLocalBasis(test_spaces, test_indices, 0);
  RetrieveLocalBasis(ansatz_spaces, ansatz_indices, test_spaces.size());

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    RetrieveLocalBasis(aux_spaces[m], aux_indices[m], aux_basis_offset[m]);
  }

  return qf_orig;
}

template<int d>
void ReducedAssembling<d>::AssembleLocalTestBasis(int c, int p)
{
  const unsigned int n_test_values = test_derivatives.size();

  // retrieve and assemble test basis values

  int offset = 0;
  for (int i = 0; i < test_vector_dim; i++)
  {
    const int* cell_dof = test_spaces[i]->GetGlobalDOF(c);
    const int n_cell_dof = test_spaces[i]->get_n_local_dof(c);

    for (unsigned int r = 0; r < n_test_values; r++)
    {
      const double* bv = orig_basis_values[i][test_derivative_lookup[r]][p];

      auto slice = test_values.slice(2, { (unsigned)i, r, 0 }, false);

      for (int j = 0; j < test_dim; j++)
      {
        double &v = slice[j];

        v = 0.0;
        for (int k = 0; k < n_cell_dof; k++)
        {
          v += test_basis[cell_dof[k] + offset][j] * bv[k];
        }
      }
    }

    offset += test_spaces[i]->get_n_dof();
  }
}

template<int d>
void ReducedAssembling<d>::AssembleLocalAnsatzBasis(int c, int p)
{
  const unsigned int n_ansatz_values = ansatz_derivatives.size();

  // retrieve and assemble ansatz basis values

  int offset = 0;
  for (int i = 0; i < ansatz_vector_dim; i++)
  {
    const int* cell_dof = ansatz_spaces[i]->GetGlobalDOF(c);
    const int n_cell_dof = ansatz_spaces[i]->get_n_local_dof(c);

    for (unsigned int r = 0; r < n_ansatz_values; r++)
    {
      const double* bv = orig_basis_values[test_vector_dim + i][ansatz_derivative_lookup[r]][p];

      auto slice = ansatz_values.slice(2, { (unsigned)i, r, 0 }, false);

      for (int j = 0; j < ansatz_dim; j++)
      {
        double &v = slice[j];

        v = 0.0;
        for (int k = 0; k < n_cell_dof; k++)
        {
          v += ansatz_basis[cell_dof[k] + offset][j] * bv[k];
        }
      }
    }

    offset += ansatz_spaces[i]->get_n_dof();
  }
}

template<int d>
void ReducedAssembling<d>::AssembleLocalReducedFunctions()
{
  // assemble reduced FE function values

  for (unsigned int i = 0; i < reduced_functions.size(); i++)
  {
    const FunctionInfo& f = reduced_functions[i];

    const int vector_dim = f.is_test ? test_vector_dim : ansatz_vector_dim;
    const int dim = f.is_test ? test_dim : ansatz_dim;

    for (unsigned int r = 0; r < f.derivatives.size(); r++)
    {
      const int dd = reduced_derivative_lookup[i][r];
      auto fv = reduced_values[i].slice(0, { 0, r }, false);

      for (int j = 0; j < vector_dim; j++)
      {
        const auto values = (f.is_test ? test_values : ansatz_values)
          .slice(2, { (unsigned)j, (unsigned)dd, 0 });

        double &v = fv[j];

        v = 0.0;
        for (int k = 0; k < dim; k++)
        {
          v += values[k] * f.func[k];
        }
      }
    }
  }
}

template<int d>
void ReducedAssembling<d>::AssembleLocalFullFunctions(int c, int p)
{
  // assemble full FE function values

  for (unsigned int i = 0; i < full_functions.size(); i++)
  {
    const FunctionInfo& f = full_functions[i];
    const double* fc = f.func;

    const auto& spaces = f.is_test ? test_spaces : ansatz_spaces;
    const int vector_dim = (int)spaces.size();
    const int offset = f.is_test ? 0 : test_vector_dim;

    for (unsigned int r = 0; r < f.derivatives.size(); r++)
    {
      const int dd = full_derivative_lookup[i][r];
      auto fv = full_values[i].slice(0, { 0, r }, false);

      int vector_offset = 0;

      for (int j = 0; j < vector_dim; j++)
      {
        const int* cell_dof = spaces[j]->GetGlobalDOF(c);
        const int n_cell_dof = spaces[j]->get_n_local_dof(c);

        const double* bv = orig_basis_values[offset + j][dd][p];

        double &v = fv[j];

        v = 0.0;
        for (int k = 0; k < n_cell_dof; k++)
        {
          v += fc[cell_dof[k] + vector_offset] * bv[k];
        }

        vector_offset += spaces[j]->get_n_dof();
      }
    }
  }
}

template<int d>
void ReducedAssembling<d>::AssembleLocalAuxFunctions(int c, int p)
{
  // assemble auxiliary FE function values

  int aux_offset = 0;

  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    const auto& spaces = aux_spaces[m];
    const int vector_dim = (int)spaces.size();
    const unsigned int offset = aux_basis_offset[m];

    for (unsigned int i = 0; i < aux_functions[m].size(); i++)
    {
      const FunctionInfo& f = aux_functions[m][i];
      const double* fc = f.func;

      for (unsigned int r = 0; r < f.derivatives.size(); r++)
      {
        const int dd = aux_derivative_lookup[m][i][r];
        auto fv = aux_values[aux_offset].slice(0, { 0, r }, false);

        int vector_offset = 0;

        for (int j = 0; j < vector_dim; j++)
        {
          const int* cell_dof = spaces[j]->GetGlobalDOF(c);
          const int n_cell_dof = spaces[j]->get_n_local_dof(c);

          const double* bv = orig_basis_values[offset + j][dd][p];
          double &v = fv[j];

          v = 0.0;
          for (int k = 0; k < n_cell_dof; k++)
          {
            v += fc[cell_dof[k] + vector_offset] * bv[k];
          }

          vector_offset += spaces[j]->get_n_dof();
        }
      }

      aux_offset++;
    }
  }
}

template<int d>
void ReducedAssembling<d>::Assemble(double multiplier,
  double** matrix, double* rhs, bool clear, double* val)
{
  #ifdef TIME_REDUCED_ASSEMBLING
  Chrono t_retrieval, t_gather, t_assemble, t_total;

  t_retrieval.reset();
  t_gather.reset();
  t_assemble.reset();
  #endif

  if (clear)
  {
    if (matrix != nullptr)
    {
      for (int i = 0; i < ansatz_dim; i++)
      {
        memset(matrix[i], 0, test_dim * sizeof(double));
      }
    }

    if (rhs != nullptr)
    {
      memset(rhs, 0, test_dim * sizeof(double));
    }
  }

  bool distinct_bases = DistinctBases();

  if (!setup_finished)
  {
    SetBasisAndFunctionStorage();
    SetDerivativeLookups();

    setup_finished = true;
  }

  unsigned int sd_size = test_spaces.size() + ansatz_spaces.size();
  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    sd_size += aux_spaces[m].size();
  }

  orig_basis_values.resize(sd_size);
  elements.resize(sd_size);
  bf.resize(sd_size);

  // bool* because you can't get the data() of a bool vector
  bool* sd = new bool[sd_size];

  for (unsigned int i = 0; i < test_spaces.size(); i++)
  {
    sd[i] = test_sd;
  }

  for (unsigned int i = 0; i < ansatz_spaces.size(); i++)
  {
    sd[test_spaces.size() + i] = ansatz_sd;
  }

  unsigned int sd_offset = test_spaces.size() + ansatz_spaces.size();
  for (unsigned int m = 0; m < aux_spaces.size(); m++)
  {
    aux_basis_offset[m] = sd_offset;

    for (unsigned int i = 0; i < aux_spaces[m].size(); i++)
    {
      sd[sd_offset++] = aux_sd[m];
    }
  }

  ReducedAssemblingData data =
  {
    matrix,
    rhs,
    val,

    test_dim,
    ansatz_dim,

    test_values,
    distinct_bases ? ansatz_values : test_values,
    reduced_values,
    full_values,
    aux_values
  };

  int n_assembly_functions = (int)assembly_functions.size();

  auto collection = test_spaces[0]->GetCollection();

  int n_cells = collection->GetN_Cells();

  for (int c = 0; c < n_cells; c++)
  {
    #ifdef TIME_REDUCED_ASSEMBLING
    t_retrieval.start();
    #endif

    const TBaseCell* cell = collection->GetCell(c);
    const double hK = cell->Get_hK(TDatabase::ParamDB->CELL_MEASURE);

    // retrieve local FE information
    TQuadFormula qf = RetrieveLocalFEData(collection, c, sd);

    int n_points = qf.GetN_QuadPoints();

    #ifdef TIME_REDUCED_ASSEMBLING
    t_retrieval.stop();
    #endif

    for (int p = 0; p < n_points; p++)
    {
      #ifdef TIME_REDUCED_ASSEMBLING
      t_gather.start();
      #endif

      AssembleLocalTestBasis(c, p);

      if (distinct_bases)
      {
        AssembleLocalAnsatzBasis(c, p);
      }

      AssembleLocalReducedFunctions();
      AssembleLocalFullFunctions(c, p);
      AssembleLocalAuxFunctions(c, p);

      #ifdef TIME_REDUCED_ASSEMBLING
      t_gather.stop();

      t_assemble.start();
      #endif

      // call assembly routines

      double weight = multiplier * qf.get_weight(p);

      for (int f = 0; f < n_assembly_functions; f++)
      {
        assembly_functions[f](weight * assembly_multipliers[f], hK, data);
      }

      #ifdef TIME_REDUCED_ASSEMBLING
      t_assemble.stop();
      #endif
    }
  }

  delete[] sd;

  #ifdef TIME_REDUCED_ASSEMBLING
  t_total.stop();

  Output::root_info("ReducedAssembling", "Assemble total time: ", t_total.elapsed_time(),
    "\nRetrieval: ", t_retrieval.elapsed_time(), ", ", 100 * t_retrieval.elapsed_time() / t_total.elapsed_time(), "%",
    "\nGather: ", t_gather.elapsed_time(), ", ", 100 * t_gather.elapsed_time() / t_total.elapsed_time(), "%",
    "\nAssemble: ", t_assemble.elapsed_time(), ", ", 100 * t_assemble.elapsed_time() / t_total.elapsed_time(), "%");
  #endif
}

namespace ReducedBasisLookup
{
  /// @brief If the basis is in default order, returns v_k
  const Slice<double> DefaultBasisValues(const MultiVector<double, 3>& basis, int k)
  {
    return basis.slice(2, { (size_t)k, 0, 0 });
  }

  /// @brief If the basis is in default order, returns d/dx_m v_k
  const Slice<double> DefaultBasisGradients(const MultiVector<double, 3>& basis, int k, int m)
  {
    return basis.slice(2, { (size_t)k, (size_t)(1 + m), 0 });
  }

  /// @brief If the basis is in default order, returns d/dx_m v_0
  const Slice<double> DefaultBasisGradients(const MultiVector<double, 3>& basis, int m)
  {
    return basis.slice(2, { 0, (size_t)(1 + m), 0 });
  }

  /// @brief If the function is in default order, returns u_k
  double DefaultFEValue(const MultiVector<double, 2>& func, int k)
  {
    return func[{ (size_t)k, 0 }];
  }

  /// @brief If the function is in default order, returns d/dx_m u_k
  double DefaultFEGradient(const MultiVector<double, 2>& func, int k, int m)
  {
    return func[{ (size_t)k, (size_t)(1 + m) }];
  }

  /// @brief If the function is in default order, returns d/dx_m u_0
  double DefaultFEGradient(const MultiVector<double, 2>& func, int m)
  {
    return func[{ 0, (size_t)(1 + m) }];
  }
}

#ifdef __3D__
template class ReducedAssembling<3>;
#endif

#ifdef __2D__
template class ReducedAssembling<2>;
#endif