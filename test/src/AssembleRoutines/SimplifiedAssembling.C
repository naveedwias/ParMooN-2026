#include <SimplifiedAssembling.h>
#include <Database.h>
#include <FEDatabase.h>
#include <BaseCell.h>
#include <MooNMD_Io.h>

#include <BlockFEMatrix.h>
#include <BlockVector.h>

#ifdef __3D__
#include <FESpace3D.h>
#endif

#ifdef __2D__
#include <FESpace2D.h>
#endif

#include <algorithm>
#include <memory>
#include <string.h>

template<int d>
SimplifiedAssembling<d>::SimplifiedAssembling(
  std::vector<std::shared_ptr<const FESpace>> test_spaces,
  std::vector<std::shared_ptr<const FESpace>> ansatz_spaces,
  MultiIndex_vector test_derivatives,
  MultiIndex_vector ansatz_derivatives)
: test_spaces(test_spaces), ansatz_spaces(ansatz_spaces),
  test_vector_dim((int)test_spaces.size()), ansatz_vector_dim((int)ansatz_spaces.size()),
  test_derivatives(test_derivatives), ansatz_derivatives(ansatz_derivatives),

  setup_finished(false)
{
  if (test_vector_dim == 0)
  {
    ErrThrow("No test spaces provided")
  }

  if (ansatz_vector_dim == 0)
  {
    ErrThrow("No ansatz spaces provided")
  }

#ifdef _MPI
  Output::root_warn("SimplifiedAssembling", "Simplified assembly may not "
    "function correctly in the parallelized case.");
#endif
}

template<int d>
SimplifiedAssembling<d>::SimplifiedAssembling(
  std::vector<std::shared_ptr<const FESpace>> spaces,
  MultiIndex_vector derivatives)
: SimplifiedAssembling(spaces, spaces,
  derivatives, derivatives)
{
}

template<int d>
SimplifiedAssembling<d>::FunctionInfo::FunctionInfo(const double* func,
  const MultiIndex_vector& derivatives,
  const bool is_test)
: func(func), derivatives(derivatives), is_test(is_test)
{
}

template<int d>
SimplifiedAssembling<d>::AssemblyInfo::AssemblyInfo(AssembleLocalSimplified func, int row, int column, float multiplier)
: func(func), row(row), column(column), multiplier(multiplier)
{
}

template<int d>
void SimplifiedAssembling<d>::AddAssembly(AssembleLocalSimplified assembly, int r, int c, double m)
{
  if (r < 0 || r >= test_vector_dim || c < 0 || c >= ansatz_vector_dim)
  {
    ErrThrow("Bad assembly block ", r, ", ", c);
  }

  assembly_functions.emplace_back(assembly, r, c, m);

  setup_finished = false;
}

template<int d>
void SimplifiedAssembling<d>::AddFEFunction(const double* func,
  const MultiIndex_vector& derivatives, bool is_test)
{
  fe_functions.emplace_back(func, derivatives, is_test || !DistinctBases());

  setup_finished = false;
}

template<int d>
void SimplifiedAssembling<d>::Reset()
{
  assembly_functions.clear();
  fe_functions.clear();

  setup_finished = false;
}

template<int d>
bool SimplifiedAssembling<d>::DistinctBases()
{
  // check whether test and ansatz spaces or base data are distinct

  int n_test_values = (int)test_derivatives.size();
  int n_ansatz_values = (int)ansatz_derivatives.size();

  if ((test_vector_dim != ansatz_vector_dim)
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
void SimplifiedAssembling<d>::SetBasisAndFunctionStorage()
{
  unsigned int n_test_values = test_derivatives.size();
  unsigned int n_ansatz_values = ansatz_derivatives.size();

  // check whether test and ansatz spaces or base data are distinct
  bool distinct_bases = DistinctBases();

  // set up vectors to hold basis values

  test_values.resize((unsigned)test_vector_dim);
  for (int i = 0; i < test_vector_dim; i++)
  {
    test_values[i].resize({ n_test_values, test_spaces[i]->get_max_n_local_dof() });
  }

  if (distinct_bases)
  {
    ansatz_values.resize((unsigned)ansatz_vector_dim);
    for (int i = 0; i < ansatz_vector_dim; i++)
    {
      ansatz_values[i].resize({ n_ansatz_values, ansatz_spaces[i]->get_max_n_local_dof() });
    }
  }

  // set up vectors to hold FE function values

  fe_values.resize(fe_functions.size());
  for (unsigned int i = 0; i < fe_functions.size(); i++)
  {
    fe_values[i].resize(
      {
        (unsigned)(fe_functions[i].is_test ? test_vector_dim : ansatz_vector_dim),
        fe_functions[i].derivatives.size()
      });
  }
}

template<int d>
void SimplifiedAssembling<d>::SetDerivativeLookup(const MultiIndex_vector& derivatives,
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
void SimplifiedAssembling<d>::SetDerivativeLookups()
{
  test_sd = needs_second_derivatives<d>(test_derivatives);
  ansatz_sd = needs_second_derivatives<d>(ansatz_derivatives);

  for (const FunctionInfo &f: fe_functions)
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

  // create lookup structures to map derivative order to retrieved order

  test_indices = indices_up_to_order<d>(test_sd ? 2 : 1);
  ansatz_indices = indices_up_to_order<d>(ansatz_sd ? 2 : 1);

  SetDerivativeLookup(test_derivatives, test_indices, test_derivative_lookup);
  SetDerivativeLookup(ansatz_derivatives, ansatz_indices, ansatz_derivative_lookup);

  fe_derivative_lookup.resize(fe_functions.size());

  for (unsigned int i = 0; i < fe_functions.size(); i++)
  {
    const FunctionInfo& f = fe_functions[i];

    if (f.is_test)
    {
      SetDerivativeLookup(f.derivatives, test_derivatives, fe_derivative_lookup[i]);
    }
    else
    {
      SetDerivativeLookup(f.derivatives, ansatz_derivatives, fe_derivative_lookup[i]);
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::RetrieveLocalElements(const std::vector<std::shared_ptr<const FESpace>>& spaces,
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
void SimplifiedAssembling<d>::RetrieveLocalBasis(const std::vector<std::shared_ptr<const FESpace>>& spaces,
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
TQuadFormula SimplifiedAssembling<d>::RetrieveLocalFEData(
  const TCollection* collection, int c, bool* sd)
{
  const TBaseCell* cell = collection->GetCell(c);

  // retrieve local FE information

  RetrieveLocalElements(test_spaces, c, 0);
  RetrieveLocalElements(ansatz_spaces, c, test_vector_dim);

  TQuadFormula qf_ref(QuadratureFormula_type::BaryCenterTetra);
  TQuadFormula qf_orig(qf_ref);

  // retrieve common quadrature formula and set cell
  // (FEDatabase has ugly hidden state)

  FEDatabase::GetOrig(
    elements, collection, cell, sd,
    qf_ref, qf_orig);

  // retrieve local full-order FE bases

  RetrieveLocalBasis(test_spaces, test_indices, 0);
  RetrieveLocalBasis(ansatz_spaces, ansatz_indices, test_vector_dim);

  return qf_orig;
}

template<int d>
void SimplifiedAssembling<d>::AssembleLocalTestBasis(int c, int p)
{
  unsigned int n_test_values = test_derivatives.size();

  // retrieve and assemble test basis values

  for (int i = 0; i < test_vector_dim; i++)
  {
    int test_dim = test_spaces[i]->get_max_n_local_dof();

    const int n_cell_dof = test_spaces[i]->get_n_local_dof(c);

    for (unsigned int r = 0; r < n_test_values; r++)
    {
      double* bv = orig_basis_values[i][test_derivative_lookup[r]][p];
      auto slice = test_values[i].slice(1, { r, 0 }, false);

      for (int j = 0; j < n_cell_dof; j++)
      {
        slice[j] = bv[j];
      }

      for (int j = n_cell_dof; j < test_dim; j++)
      {
        slice[j] = 0.0;
      }
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::AssembleLocalAnsatzBasis(int c, int p)
{
  unsigned int n_ansatz_values = ansatz_derivatives.size();

  // retrieve and assemble ansatz basis values

  for (int i = 0; i < ansatz_vector_dim; i++)
  {
    int ansatz_dim = ansatz_spaces[i]->get_max_n_local_dof();

    const int n_cell_dof = ansatz_spaces[i]->get_n_local_dof(c);

    for (unsigned int r = 0; r < n_ansatz_values; r++)
    {
      double* bv = orig_basis_values[test_vector_dim + i][ansatz_derivative_lookup[r]][p];
      auto slice = ansatz_values[i].slice(1, { r, 0 }, false);

      for (int j = 0; j < n_cell_dof; j++)
      {
        slice[j] = bv[j];
      }

      for (int j = n_cell_dof; j < ansatz_dim; j++)
      {
        slice[j] = 0.0;
      }
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::AssembleLocalFEFunctions(int c, int p)
{
  // assemble FE function values

  for (unsigned int i = 0; i <  fe_functions.size(); i++)
  {
    const FunctionInfo& f = fe_functions[i];
    const double* fc = f.func;

    const auto& spaces = f.is_test ? test_spaces : ansatz_spaces;
    const int vector_dim = (int)spaces.size();
    const int offset = f.is_test ? 0 : test_vector_dim;

    for (unsigned int r = 0; r < f.derivatives.size(); r++)
    {
      const int dd = fe_derivative_lookup[i][r];
      auto fv = fe_values[i].slice(0, { 0, r }, false);

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
SimplifiedAssembling<d>::AssemblyLoop::AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, BlockFEMatrix* mat, BlockVector *rhs, bool clear)
: AssemblyLoop(assembler, multiplier, true, mat != nullptr, rhs != nullptr)
{
  if (has_matrix)
  {
    // note that the matrix is allowed to be larger
    if (mat->get_n_cell_columns() < (unsigned)assembler.ansatz_vector_dim || mat->get_n_cell_rows() < (unsigned)assembler.test_vector_dim)
    {
      ErrThrow("Provided matrix is too small!");
    }

    for (int r = 0; r < assembler.test_vector_dim; r++)
    {
      if (mat->get_test_space((unsigned)r, 0) != assembler.test_spaces[r])
      {
        ErrThrow("Matrix row spaces dont match test spaces!");
      }
    }

    for (int c = 0; c < assembler.ansatz_vector_dim; c++)
    {
      if (mat->get_ansatz_space(0, (unsigned)c) != assembler.ansatz_spaces[c])
      {
        ErrThrow("Matrix column spaces dont match ansatz spaces!");
      }
    }
  }

  if (has_rhs)
  {
    if (rhs->n_blocks() < (unsigned)assembler.test_vector_dim)
    {
      ErrThrow("Provided rhs vector is too small!");
    }

    for (int r = 0; r < assembler.test_vector_dim; r++)
    {
      if ((int)rhs->length(r) != assembler.test_spaces[r]->get_n_dof())
      {
        ErrThrow("RHS block lengths don't match test spaces!");
      }
    }
  }

  if (clear)
  {
    if (has_matrix)
    {
      mat->reset();
    }

    if (has_rhs)
    {
      rhs->reset();
    }
  }

  block_fe_mat = mat;
  block_rhs = rhs;
}

template<int d>
SimplifiedAssembling<d>::AssemblyLoop::AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, FEMatrix* mat, double *rhs, bool clear)
: AssemblyLoop(assembler, multiplier, false, mat != nullptr, rhs != nullptr)
{
  if (has_matrix)
  {
    if (assembler.test_spaces.size() != 1 || assembler.ansatz_spaces.size() != 1)
    {
      ErrThrow("Can only assemble 1x1 block matrices as FEMatrix! This "
        "assembler expects ",
        assembler.test_spaces.size(), " block rows, ",
        assembler.ansatz_spaces.size(), " block columns.");
    }

    if (assembler.test_spaces[0] != mat->GetTestSpace())
    {
      ErrThrow("Mismatched test spaces! Matrix has ",
        mat->GetTestSpace()->GetName(), " expected ",
        assembler.test_spaces[0]->GetName());
    }

    if (assembler.ansatz_spaces[0] != mat->GetAnsatzSpace())
    {
      ErrThrow("Mismatched ansatz spaces! Matrix has ",
        mat->GetAnsatzSpace()->GetName(), " expected ",
        assembler.ansatz_spaces[0]->GetName())
    }
  }

  if (clear)
  {
    if (has_matrix)
    {
      mat->reset();
    }

    if (has_rhs)
    {
      memset(rhs, 0, assembler.test_spaces[0]->get_n_dof() * sizeof(double));
    }
  }

  fe_mat = mat;
  flat_rhs = rhs;
}

template<int d>
SimplifiedAssembling<d>::AssemblyLoop::AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier)
: AssemblyLoop(assembler, multiplier, false, false, false)
{
}

template<int d>
SimplifiedAssembling<d>::AssemblyLoop::AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, bool block_type, bool has_matrix, bool has_rhs)
: assembler(assembler), multiplier(multiplier), block_type(block_type),
  has_matrix(has_matrix), has_rhs(has_rhs)
{
  distinct_bases = assembler.DistinctBases();

  if (!assembler.setup_finished)
  {
    assembler.SetBasisAndFunctionStorage();
    assembler.SetDerivativeLookups();

    assembler.setup_finished = true;
  }

  unsigned int sd_size = assembler.test_vector_dim + assembler.ansatz_vector_dim;

  assembler.orig_basis_values.resize(sd_size);
  assembler.elements.resize(sd_size);
  assembler.bf.resize(sd_size);

  // bool* because you can't get the data() of a bool vector
  sd = new bool[sd_size];

  max_row_dim = 0;
  max_col_dim = 0;

  for (int i = 0; i < assembler.test_vector_dim; i++)
  {
    sd[i] = assembler.test_sd;

    max_row_dim = std::max(max_row_dim, assembler.test_spaces[i]->get_max_n_local_dof());
  }

  for (int i = 0; i < assembler.ansatz_vector_dim; i++)
  {
    sd[assembler.test_vector_dim + i] = assembler.ansatz_sd;

    max_col_dim = std::max(max_col_dim, assembler.ansatz_spaces[i]->get_max_n_local_dof());
  }

  unsigned int local_matrix_size = max_col_dim * max_row_dim;

  local_matrix_data.resize(local_matrix_size);
  local_matrix.resize(max_col_dim);
  local_rhs.resize(max_row_dim);

  for (unsigned int c = 0; c < max_col_dim; c++)
  {
    local_matrix[c] = local_matrix_data.data() + c * max_row_dim;
  }

  int bloff = 0;
  block_offset.resize(assembler.test_vector_dim);
  cell_index = 0;

  for (int r = 0; r < assembler.test_vector_dim; r++)
  {
    int test_dim = assembler.test_spaces[r]->get_max_n_local_dof();

    for (int c = 0; c < assembler.ansatz_vector_dim; c++)
    {
      int ansatz_dim = assembler.ansatz_spaces[r]->get_max_n_local_dof();

      data.emplace_back(
        local_matrix.data(),
        local_rhs.data(),

        test_dim,
        ansatz_dim,
        assembler.test_values[r],
        distinct_bases ? assembler.ansatz_values[c] : assembler.test_values[c],
        assembler.fe_values,
        cell_index, point.data(),
        r, c);
    }

    block_offset[r] = bloff;
    bloff += test_dim;
  }
}

template<int d>
SimplifiedAssembling<d>::AssemblyLoop::~AssemblyLoop()
{
  delete[] sd;
}

template<int d>
void SimplifiedAssembling<d>::AssemblyLoop::TransferBlockData(const AssemblyInfo &info, int c)
{
  auto& r_space = assembler.test_spaces[info.row];
  auto& c_space = assembler.ansatz_spaces[info.column];

  const int* cell_dof_r = r_space->GetGlobalDOF(c);
  const int n_cell_dof_r = r_space->get_n_local_dof(c);

  const int* cell_dof_c = c_space->GetGlobalDOF(c);
  const int n_cell_dof_c = c_space->get_n_local_dof(c);

  if (has_matrix)
  {
    bool tr = false;
    auto block = block_fe_mat->get_block(info.row, info.column, tr).get();

    const TStructure &structure = block->GetStructure();
    double* block_data = block->GetEntries();

    if (tr)
    {
      for (int col = 0; col < n_cell_dof_c; col++)
      {
        for (int row = 0; row < n_cell_dof_r; row++)
        {
          block_data[structure.get_index_of_entry(cell_dof_c[col], cell_dof_r[row])]
            += local_matrix[col][row];
        }
      }
    }
    else
    {
      for (int col = 0; col < n_cell_dof_c; col++)
      {
        for (int row = 0; row < n_cell_dof_r; row++)
        {
          block_data[structure.get_index_of_entry(cell_dof_r[row], cell_dof_c[col])]
            += local_matrix[col][row];
        }
      }
    }
  }

  if (has_rhs)
  {
    double* rhs_data = block_rhs->block(info.row);

    for (int i = 0; i < n_cell_dof_r; i++)
    {
      rhs_data[cell_dof_r[i]] += local_rhs[i];
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::AssemblyLoop::TransferFlatData(const AssemblyInfo &info, int c)
{
  auto& r_space = assembler.test_spaces[info.row];
  auto& c_space = assembler.ansatz_spaces[info.column];

  const int* cell_dof_r = r_space->GetGlobalDOF(c);
  const int n_cell_dof_r = r_space->get_n_local_dof(c);

  const int* cell_dof_c = c_space->GetGlobalDOF(c);
  const int n_cell_dof_c = c_space->get_n_local_dof(c);

  if (has_matrix)
  {
    const TStructure &structure = fe_mat->GetStructure();
    double* block_data = fe_mat->GetEntries();

    for (int col = 0; col < n_cell_dof_c; col++)
    {
      for (int row = 0; row < n_cell_dof_r; row++)
      {
        block_data[structure.get_index_of_entry(cell_dof_r[row], cell_dof_c[col])]
          += local_matrix[col][row];
      }
    }
  }

  if (has_rhs)
  {
    double* rhs_data = flat_rhs + block_offset[info.row];

    for (int i = 0; i < n_cell_dof_r; i++)
    {
      rhs_data[cell_dof_r[i]] += local_rhs[i];
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::AssemblyLoop::Assemble()
{
  int n_assembly_functions = (int)assembler.assembly_functions.size();

  auto collection = assembler.test_spaces[0]->GetCollection();

  int n_cells = collection->GetN_Cells();

  using namespace std::placeholders;

  auto TransferData = std::bind(block_type 
    ? &AssemblyLoop::TransferBlockData
    : &AssemblyLoop::TransferFlatData,
    this, _1, _2);

  for (int c = 0; c < n_cells; c++)
  {
    cell_index = c;

    const TBaseCell* cell = collection->GetCell(c);
    const double hK = cell->Get_hK(TDatabase::ParamDB->CELL_MEASURE);

    // retrieve local FE information
    TQuadFormula qf = assembler.RetrieveLocalFEData(collection, c, sd);

    int n_points = qf.GetN_QuadPoints();

    for (int p = 0; p < n_points; p++)
    {
      auto pt = qf.get_point((unsigned)p);
      point[0] = pt.x;
      point[1] = pt.y;
      if (d > 2)
      {
        point[2] = pt.z;
      }

      assembler.AssembleLocalTestBasis(c, p);

      if (distinct_bases)
      {
        assembler.AssembleLocalAnsatzBasis(c, p);
      }

      assembler.AssembleLocalFEFunctions(c, p);

      // call assembly routines

      double weight = multiplier * qf.get_weight(p);

      for (int f = 0; f < n_assembly_functions; f++)
      {
        memset(local_matrix_data.data(), 0, local_matrix_data.size() * sizeof(double));
        memset(local_rhs.data(), 0, max_row_dim * sizeof(double));

        const AssemblyInfo &info = assembler.assembly_functions[f];

        info.func(weight * info.multiplier, hK, data[info.row * assembler.ansatz_vector_dim + info.column]);

        TransferData(info, c);
      }
    }
  }
}

template<int d>
void SimplifiedAssembling<d>::Assemble(double multiplier,
  BlockFEMatrix *matrix, BlockVector *rhs, bool clear)
{
  AssemblyLoop(*this, multiplier, matrix, rhs, clear).Assemble();
}

template<int d>
void SimplifiedAssembling<d>::Assemble(double multiplier,
  FEMatrix *matrix, double *rhs, bool clear)
{
  AssemblyLoop(*this, multiplier, matrix, rhs, clear).Assemble();
}

template<int d>
void SimplifiedAssembling<d>::Assemble(double multiplier)
{
  AssemblyLoop(*this, multiplier).Assemble();
}

namespace SimplifiedAssemblingHelpers
{
  void MassMatrix(double w, double, const SimplifiedAssemblingData& data,
    unsigned int mr, unsigned int mc)
  {
    const auto vr = data.row_basis.slice(1, { mr, 0 });
    const auto vc = data.col_basis.slice(1, { mc, 0 });

    vc.tensor_product(data.matrix, vr, w);
  }

  void L2ProductMatrix(double w, double, const SimplifiedAssemblingData& data)
  {
    const auto vr = data.row_basis.slice(1, { 0, 0 });
    const auto vc = data.col_basis.slice(1, { 0, 0 });

    vc.tensor_product(data.matrix, vr, w);
  }

  template<int d>
  void H1ProductMatrix(double w, double, const SimplifiedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      const auto dk_vr = data.row_basis.slice(1, { 1 + (unsigned)k, 0 });
      const auto dk_vc = data.col_basis.slice(1, { 1 + (unsigned)k, 0 });

      dk_vc.tensor_product(data.matrix, dk_vr, w);
    }
  }

  void DivProductMatrix(double w, double, const SimplifiedAssemblingData& data, int k)
  {
    const auto q = data.row_basis.slice(1, { 0, 0 });
    const auto dk_vk = data.col_basis.slice(1, { 1 + (unsigned)k, 0 });

    dk_vk.tensor_product(data.matrix, q, w);
  }

  void GradProductMatrix(double w, double, const SimplifiedAssemblingData& data, int k)
  {
    const auto dk_q = data.col_basis.slice(1, { 1 + (unsigned)k, 0 });
    const auto vk = data.row_basis.slice(1, { 0, 0 });

    dk_q.tensor_product(data.matrix, vk, w);
  }

  template<int d>
  void StiffnessMatrix(double w, double, const SimplifiedAssemblingData& data,
    const std::array<unsigned int, d> &mr, const std::array<unsigned int, d> &mc)
  {
    for (int k = 0; k < d; k++)
    {
      const auto dk_vr = data.row_basis.slice(1, { mr[k], 0 });
      const auto dk_vc = data.col_basis.slice(1, { mc[k], 0 });

      dk_vc.tensor_product(data.matrix, dk_vr, w);
    }
  }

  template<int d>
  void ScaledStiffnessMatrix(double w, double hK, const SimplifiedAssemblingData& data,
    const std::array<unsigned int, d> &mr, const std::array<unsigned int, d> &mc)
  {
    w *= hK * hK;

    for (int k = 0; k < d; k++)
    {
      const auto dk_vr = data.row_basis.slice(1, { mr[k], 0 });
      const auto dk_vc = data.col_basis.slice(1, { mc[k], 0 });

      dk_vc.tensor_product(data.matrix, dk_vr, w);
    }
  }

  template<typename S, typename T>
  int get_index(T indices, S index)
  {
    for (int i = 0; i < (int)indices.size(); i++)
    {
      if (indices[i] == index)
      {
        return i;
      }
    }

    ErrThrow("Could not find index!");

    return -1;
  }
}

template<int d>
void SimplifiedAssembling<d>::AssembleMass(double multiplier,
      BlockFEMatrix *matrix,
      bool clear)
{
  using namespace SimplifiedAssemblingHelpers;
  using namespace std::placeholders;

  Reset();

  auto index = indices_of_order<d>(0)[0];

  int mr = get_index(test_derivatives, index);
  int mc = get_index(ansatz_derivatives, index);

  for (int br = 0; br < test_vector_dim; br++)
  {
    for (int bc = 0; bc < ansatz_vector_dim; bc++)
    {
      AddAssembly(std::bind(MassMatrix,
        _1, _2, _3, (unsigned)mr, (unsigned)mc), br, bc);
    }
  }

  Assemble(multiplier, matrix, nullptr, clear);
}

template<int d>
void SimplifiedAssembling<d>::AssembleStiffness(double multiplier,
      BlockFEMatrix *matrix,
      bool clear,
      bool scaled)
{
  using namespace SimplifiedAssemblingHelpers;
  using namespace std::placeholders;

  Reset();

  auto indices = indices_of_order<d>(1);

  std::array<unsigned int, d> mr;
  std::array<unsigned int, d> mc;

  for (int k = 0; k < d; k++)
  {
    mr[k] = get_index(test_derivatives, indices[k]);
    mc[k] = get_index(ansatz_derivatives, indices[k]);
  }

  for (int br = 0; br < test_vector_dim; br++)
  {
    for (int bc = 0; bc < ansatz_vector_dim; bc++)
    {
      AddAssembly(std::bind(scaled ? ScaledStiffnessMatrix<d> : StiffnessMatrix<d>,
        _1, _2, _3, mr, mc), br, bc);
    }
  }

  Assemble(multiplier, matrix, nullptr, clear);
}

template<int d>
std::shared_ptr<FEMatrix> SimplifiedAssembling<d>::AssembleMass(std::shared_ptr<const FESpace> space, double m)
{
  auto M = std::make_shared<FEMatrix>(space, space);

  using namespace SimplifiedAssemblingHelpers;

  SimplifiedAssembling<d> assembler({ space }, indices_up_to_order<d>(0));

  assembler.AddAssembly(L2ProductMatrix, 0, 0);
  assembler.Assemble(m, M.get(), nullptr);

  return std::move(M);
}

template<int d>
std::shared_ptr<FEMatrix> SimplifiedAssembling<d>::AssembleStiffness(std::shared_ptr<const FESpace> space,
  double m, bool scaled)
{
  auto A = std::make_shared<FEMatrix>(space, space);

  using namespace SimplifiedAssemblingHelpers;
  using namespace std::placeholders;

  std::array<unsigned int, d> mr;
  std::array<unsigned int, d> mc;

  for (int k = 0; k < d; k++)
  {
    mr[k] = mc[k] = 1 + (unsigned)k;
  }

  SimplifiedAssembling<d> assembler({ space }, indices_up_to_order<d>(1));

  assembler.AddAssembly(std::bind(scaled ? ScaledStiffnessMatrix<d> : StiffnessMatrix<d>,
    _1, _2, _3, mr, mc), 0, 0);
  assembler.Assemble(m, A.get(), nullptr);

  return std::move(A);
}

template<int d>
std::shared_ptr<FEMatrix> SimplifiedAssembling<d>::AssembleL2Product(
  std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
  double m)
{
  auto B = std::make_shared<FEMatrix>(test_space, ansatz_space);

  using namespace SimplifiedAssemblingHelpers;

  SimplifiedAssembling<d> assembler({ test_space }, { ansatz_space },
    indices_up_to_order<d>(0), indices_up_to_order<d>(0));

  assembler.AddAssembly(L2ProductMatrix, 0, 0);
  assembler.Assemble(m, B.get(), nullptr);

  return std::move(B);
}

template<int d>
std::shared_ptr<FEMatrix> SimplifiedAssembling<d>::AssembleH1Product(
  std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
  double m)
{
  auto B = std::make_shared<FEMatrix>(test_space, ansatz_space);

  using namespace SimplifiedAssemblingHelpers;

  SimplifiedAssembling<d> assembler({ test_space }, { ansatz_space },
    indices_up_to_order<d>(0), indices_up_to_order<d>(0));

  assembler.AddAssembly(H1ProductMatrix<d>, 0, 0);
  assembler.Assemble(m, B.get(), nullptr);

  return std::move(B);
}

template<int d>
std::vector<std::shared_ptr<FEMatrix>> SimplifiedAssembling<d>::AssembleDivProduct(
  std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
  double m)
{
  std::vector<std::shared_ptr<FEMatrix>> D(d, nullptr);

  using namespace SimplifiedAssemblingHelpers;
  using namespace std::placeholders;

  SimplifiedAssembling<d> assembler({ test_space }, { ansatz_space },
    indices_up_to_order<d>(0), indices_up_to_order<d>(1));

  for (int k = 0; k < d; k++)
  {
    D[k] = std::make_shared<FEMatrix>(test_space, ansatz_space);

    assembler.Reset();
    assembler.AddAssembly(std::bind(DivProductMatrix,
      _1, _2, _3, k), 0, 0);
    assembler.Assemble(m, D[k].get(), nullptr);
  }

  return std::move(D);
}

template<int d>
std::vector<std::shared_ptr<FEMatrix>> SimplifiedAssembling<d>::AssembleGradProduct(
  std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
  double m)
{
  std::vector<std::shared_ptr<FEMatrix>> D(d, nullptr);

  using namespace SimplifiedAssemblingHelpers;
  using namespace std::placeholders;

  SimplifiedAssembling<d> assembler({ test_space }, { ansatz_space },
    indices_up_to_order<d>(0), indices_up_to_order<d>(1));

  for (int k = 0; k < d; k++)
  {
    D[k] = std::make_shared<FEMatrix>(test_space, ansatz_space);

    assembler.Reset();
    assembler.AddAssembly(std::bind(GradProductMatrix,
      _1, _2, _3, k), 0, 0);
    assembler.Assemble(m, D[k].get(), nullptr);
  }

  return std::move(D);
}

SimplifiedAssemblingData::SimplifiedAssemblingData(double** matrix, double* rhs,
    int n_rows, int n_cols,
    const MultiVector<double, 2>& row_basis,
    const MultiVector<double, 2>& col_basis,
    const std::vector<MultiVector<double, 2>>& fe_func,
    const int& cell_index, const double* point,
    const int block_row, const int block_col)
: matrix(matrix), rhs(rhs),
  n_rows(n_rows), n_cols(n_cols),
  row_basis(row_basis), col_basis(col_basis),
  fe_func(fe_func),
  cell_index(cell_index), point(point),
  block_row(block_row), block_col(block_col)
{
}

#ifdef __3D__
template class SimplifiedAssembling<3>;
#endif

#ifdef __2D__
template class SimplifiedAssembling<2>;
#endif