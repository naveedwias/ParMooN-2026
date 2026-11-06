#include <MooNMD_Io.h>
#include <Preconditioner_sparse_approximate_inverse.h>
#include <BlockFEMatrix.h>
#include <BlockVector.h>
#include <SparseVector.h>
#include <CompositeOperator.h>
#include <DenseMatrix.h>

#include <string.h>
#include <unordered_map>

template <class LinearOperator, class Vector>
Preconditioner_sparse_approximate_inverse<LinearOperator, Vector>::Preconditioner_sparse_approximate_inverse(
  const LinearOperator &A, bool symmetric, bool active_rows)
: A(A), symmetric(symmetric), active_rows(active_rows)
{
  #ifdef _MPI
  Output::warn("SAI", "SAI preconditioner is not currently reliably MPI-compatible!");
  #endif

  update();
}

template<>
void Preconditioner_sparse_approximate_inverse<BlockFEMatrix, BlockVector>::update()
{
  unsigned int n_cell_rows = A.get_n_cell_rows();
  unsigned int n_cell_cols = A.get_n_cell_columns();

  Output::info<4>("SAI", "Creating temp blocks...");

  if (M == nullptr)
  {
    // copy structure
    M = std::make_shared<BlockFEMatrix>(A);
  }

  // make duplicate blocks for reinsertion at the end
  std::vector<std::shared_ptr<FEMatrix>> blocks(n_cell_rows * n_cell_cols);
  std::vector<bool> blocks_t(n_cell_rows * n_cell_cols);
  for (unsigned int r = 0; r < n_cell_rows; r++)
  {
    for (unsigned int c = 0; c < n_cell_cols; c++)
    {
      unsigned int i = r * n_cell_cols + c;

      bool t = false;

      auto block = A.get_block(r, c, t);
      blocks_t[i] = t;

      if (block != nullptr)
      {
        blocks[i] = std::make_shared<FEMatrix>(*block);
      }
    }
  }

  // size of the matrix
  unsigned int n = A.get_n_total_rows();

  // sparse row vectors
  std::vector<SparseVector> rows;
  rows.reserve(n);

  // sparse column vectors
  std::vector<SparseVector> columns;
  columns.reserve(symmetric ? 0 : n);

  std::vector<bool> active_row(n);
  for (unsigned int i = 0; i < n; i++)
  {
    active_row[i] = active_rows
      ? LinearOperatorActiveTest::is_active_row(A, i)
      : true;
  }

  // temp arrays for sparse row and column retrieval
  std::vector<int> indices;
  std::vector<double> values;

  Output::info<4>("SAI", "Retrieving rows and columns...");

  // retrieve sparse row and column vectors
  if (symmetric)
  {
    for (unsigned int i = 0; i < n; i++)
    {
      A.get_row(i, indices, values);
      rows.emplace_back((int)n, indices, values);

      // no need to retrieve columns in the symmetric case
    }
  }
  else
  {
    for (unsigned int i = 0; i < n; i++)
    {
      A.get_row(i, indices, values);
      if (!active_row[i])
      {
        for (unsigned int j = 0; j < indices.size(); j++)
        {
          values[j] = (indices[j] == (int)i) ? 1.0 : 0.0;
        }
      }
      rows.emplace_back((int)n, indices, values, true);

      A.get_column(i, indices, values);
      for (unsigned int j = 0; j < indices.size(); j++)
      {
        if (!active_row[indices[j]])
        {
          values[j] = (indices[j] == (int)i) ? 1.0 : 0.0;
        }
      }
      columns.emplace_back((int)n, indices, values, true);
    }
  }

  int max_neighbors = 0;
  if (symmetric)
  {
    for (unsigned int r = 0; r < n; r++)
    {
      max_neighbors = std::max(max_neighbors, rows[r].get_n_entries());
    }
  }
  else
  {
    for (unsigned int r = 0; r < n; r++)
    {
      max_neighbors = std::max(max_neighbors,
        std::max(rows[r].get_n_entries(), columns[r].get_n_entries()));
    }
  }

  std::vector<double> temp_mat_data(max_neighbors * max_neighbors);
  std::vector<double*> temp_mat(max_neighbors);
  for (int i = 0 ; i < max_neighbors; i++)
  {
    temp_mat[i] = temp_mat_data.data() + max_neighbors * i;
  }

  Output::info<4>("SAI", "Computing SAI...");

  std::unordered_map<int, double> xtx_entries;
  xtx_entries.reserve((unsigned)(n * std::sqrt(n)));
  unsigned int multi_count = 0;

  auto& prod_vec = symmetric ? rows : columns;
  auto get_product = [&] (int i, int j) -> double
  {
    int k = i * n + j;

    const auto it = xtx_entries.find(k);

    if (it != xtx_entries.end())
    {
      ++multi_count;
      return it->second;
    }
    else
    {
      return xtx_entries[k] = dot(prod_vec[i], prod_vec[j]);
    }
  };

  // compute SAI columns
  int singular_columns = 0;
  std::vector<double> col_values;
  for (unsigned int c = 0; c < n; c++)
  {
    // get cell column and local column within cell
    unsigned int cell_col = 0;
    int local_col = c;
    int block_cols = (int)A.get_n_columns_in_cell(0, cell_col);
    while (local_col >= block_cols)
    {
      local_col -= block_cols;
      block_cols = (int)A.get_n_columns_in_cell(0, ++cell_col);
    }

    // retrieve neighboring DOF for column c
    // (the sparsity structure is assumed to be symmetric)
    const auto& neighbors = rows[c].get_columns();
    int n_neighbors = (int)neighbors.size();

    // Set up and solve the least squares problem for this column.

    // Compute the matrix X^T X
    for (int i = 0; i < n_neighbors; i++)
    {
      for (int j = i; j < n_neighbors; j++)
      {
        double d = get_product(neighbors[i], neighbors[j]);
        temp_mat[i][j] = d;
        temp_mat[j][i] = d;
      }
    }

    DenseMatrix XTX(n_neighbors, n_neighbors);
    for (int i = 0; i < n_neighbors; i++)
    {
      XTX.setRow(i, temp_mat[i]);
    }

    // Compute the RHS
    col_values.resize(n_neighbors);
    memcpy(col_values.data(), rows[c].get_values().data(), n_neighbors * sizeof(double));

    // Solve the least squares problem
    try
    {
      XTX.decomposeLU();
      XTX.solve(col_values.data());
    }
    catch (...)
    {
      ++singular_columns;
      memset(col_values.data(), 0, n_neighbors * sizeof(double));
      int k = rows[c].get_index(c);
      if (k >= 0)
      {
        col_values[k] = 1.0 / A.get(c, c);
      }
    }

    // Write the sparse column back into the relevant blocks
    for (int i = 0; i < n_neighbors; i++)
    {
      int row = neighbors[i];

      // get cell row and local row within cell
      unsigned int cell_row = 0;
      int local_row = row;
      int block_rows = (int)A.get_n_rows_in_cell(cell_row, 0);
      while (local_row >= block_rows)
      {
        local_row -= block_rows;
        block_rows = (int)A.get_n_rows_in_cell(++cell_row, 0);
      }

      // retrieve block pointer
      unsigned int ind = cell_row * n_cell_cols + cell_col;
      bool t = blocks_t[ind];
      std::shared_ptr<FEMatrix> block = blocks[ind];

      if (block != nullptr)
      {
        // write entry
        if (t)
        {
          block->set(local_col, local_row, col_values[i]);
        }
        else
        {
          block->set(local_row, local_col, col_values[i]);
        }
      }
    }
  }

  if (singular_columns > 0)
  {
    Output::warn("SAI", "Found ", singular_columns, "/", n, " singular least squares problems");
  }

  Output::info<4>("SAI", "Writing blocks...");

  // write blocks into SAI matrix
  for (unsigned int r = 0; r < n_cell_rows; r++)
  {
    for (unsigned int c = 0; c < n_cell_cols; c++)
    {
      unsigned int i = r * n_cell_cols + c;
      M->replace_blocks(*blocks[i], { { r, c } }, { blocks_t[i] });
    }
  }

  Output::info<4>("SAI", "Done.");
}

template<>
void Preconditioner_sparse_approximate_inverse<BlockMatrix, BlockVector>::update()
{
  ErrThrow("Sparse approximate inverse is not supported for BlockMatrix");
}

template<>
void Preconditioner_sparse_approximate_inverse<CompositeOperator<BlockVector>, BlockVector>::update()
{
  ErrThrow("Sparse approximate inverse is not supported for composite operators");
}

template <class LinearOperator, class Vector>
void Preconditioner_sparse_approximate_inverse<LinearOperator, Vector>::apply(const Vector &z, Vector &r) const
{
  if (M != nullptr)
  {
    // simply multiply by SAI matrix
    M->apply(z, r);
  }
  else
  {
    r = z;
  }
}

template class Preconditioner_sparse_approximate_inverse<BlockFEMatrix, BlockVector>;
template class Preconditioner_sparse_approximate_inverse<CompositeOperator<BlockVector>, BlockVector>;

#ifndef _MPI
template class Preconditioner_sparse_approximate_inverse<BlockMatrix, BlockVector>;
#endif