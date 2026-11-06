#include <MooNMD_Io.h>
#include <Preconditioner_block_jacobi.h>
#include <BlockFEMatrix.h>
#include <BlockVector.h>
#include <CompositeOperator.h>
#include <Utilities.h>

#include <string.h>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <numeric>

template <class LinearOperator, class Vector>
Preconditioner_block_jacobi<LinearOperator, Vector>::Preconditioner_block_jacobi(
  const LinearOperator &A,
  std::string strategy, double threshold,
  int max_block_size)
: A(A), strategy(strategy), threshold(threshold), max_block_size(max_block_size)
{
  #ifdef _MPI
  Output::warn("Block-Jacobi", "Block-Jacobi preconditioner is not currently reliably MPI-compatible!");
  #endif

  update();
}

namespace BlockJacobiStrategies
{
  /// @brief Matrix lookup wrapper
  template<class LinearOperator>
  struct MatrixLookup
  {
    public:
      MatrixLookup(const LinearOperator& A)
      : A(A), diag_entries(A.get_diagonal()), n(A.get_n_total_rows())
      {
      }

      double get(int a, int b)
      {
        if (a == b)
        {
          return diag_entries[a];
        }
        else
        {
          int k = a * n + b;
          const auto it = matrixLookup.find(k);

          if (it != matrixLookup.end())
          {
            return it->second;
          }
          else
          {
            if (!LinearOperatorActiveTest::is_active_row(A, a))
            {
              return matrixLookup[k] = (a == b) ? 1.0 : 0.0;
            }

            return matrixLookup[k] = A.get((unsigned)a, (unsigned)b);
          }
        }
      }

      double get_weight(int a, int b)
      {
        if (a == b)
        {
          return 0.0;
        }
        else
        {
          int k = std::min(a, b) * n + std::max(a, b);

          const auto it = undirectedLookup.find(k);

          if (it != undirectedLookup.end())
          {
            return it->second;
          }
          else
          {
            double v = get((unsigned)a, (unsigned)b);
            double w = get((unsigned)b, (unsigned)a);

            return undirectedLookup[k] = v * v + w * w;
          }
        }
      }

    public:
      const LinearOperator& A;
      const std::vector<double> diag_entries;
      const unsigned int n;

    private:
      std::unordered_map<int, double> matrixLookup;
      std::unordered_map<int, double> undirectedLookup;
  };

  template<class LinearOperator>
  DenseMatrix DiagonalBlock(const LinearOperator& A, const std::vector<int>& dof);

  template<>
  DenseMatrix DiagonalBlock<BlockFEMatrix>(const BlockFEMatrix& A, const std::vector<int>& dof)
  {
    const int m = (int)dof.size();
    std::vector<double> temp_row(m);

    DenseMatrix M(m, m);

    int cell_row = 0;
    int cell_row_offset = 0;
    int local_rows = (int)A.get_n_rows_in_cell(0, 0);

    int last_dof = -1;

    for (int i = 0; i < m; i++)
    {
      if (dof[i] <= last_dof)
      {
        ErrThrow("DOF are not in ascending order: ", last_dof, " -> ", dof[i]);
      }

      int ii = dof[i] - cell_row_offset;

      while (ii >= local_rows)
      {
        ii -= local_rows;
        cell_row_offset += local_rows;
        local_rows = (int)A.get_n_rows_in_cell(++cell_row, 0);
      }

      if (ii >= A.get_test_space(cell_row, 0)->get_n_active())
      {
        for (int j = 0; j < m; j++)
        {
          temp_row[j] = dof[i] == dof[j] ? 1.0 : 0.0;
        }

        M.setRow(i, temp_row.data());
        last_dof = dof[i];
        continue;
      }

      int cell_col = 0;
      int cell_col_offset = 0;
      int local_cols = (int)A.get_n_columns_in_cell(cell_row, 0);

      for (int j = 0; j < m; j++)
      {
        temp_row[j] = 0.0;

        int jj = dof[j] - cell_col_offset;

        while (jj >= local_cols)
        {
          jj -= local_cols;
          cell_col_offset += local_cols;
          local_cols = (int)A.get_n_columns_in_cell(0, ++cell_col);
        }

        bool t = false;
        const auto block = A.get_block(cell_row, cell_col, t);

        if (block != nullptr)
        {
          const int* rows = block->get_row_ptr();
          const int* columns = block->get_vector_columns();
          const double* entries = block->GetEntries();

          if (t)
          {
            int m0 = rows[jj];
            int m1 = rows[jj + 1];

            for (int k = m0; k < m1; k++)
            {
              if (columns[k] == ii)
              {
                temp_row[j] = entries[k];
              }
            }
          }
          else
          {
            int m0 = rows[ii];
            int m1 = rows[ii + 1];

            for (int k = m0; k < m1; k++)
            {
              if (columns[k] == jj)
              {
                temp_row[j] = entries[k];
              }
            }
          }
        }
      }

      M.setRow(i, temp_row.data());
      last_dof = dof[i];
    }

    return M;
  }

  template<>
  DenseMatrix DiagonalBlock<BlockMatrix>(const BlockMatrix& A, const std::vector<int>& dof)
  {
    const int m = (int)dof.size();
    std::vector<double> temp_row(m);

    DenseMatrix M(m, m);

    int cell_row = 0;
    int cell_row_offset = 0;
    int local_rows = (int)A.get_n_rows_in_cell(0, 0);

    int last_dof = -1;

    for (int i = 0; i < m; i++)
    {
      if (dof[i] <= last_dof)
      {
        ErrThrow("DOF are not in ascending order: ", last_dof, " -> ", dof[i]);
      }

      int ii = dof[i] - cell_row_offset;

      while (ii >= local_rows)
      {
        ii -= local_rows;
        cell_row_offset += local_rows;
        local_rows = (int)A.get_n_rows_in_cell(++cell_row, 0);
      }

      int cell_col = 0;
      int cell_col_offset = 0;
      int local_cols = (int)A.get_n_columns_in_cell(cell_row, 0);

      for (int j = 0; j < m; j++)
      {
        temp_row[j] = 0.0;

        int jj = dof[j] - cell_col_offset;

        while (jj >= local_cols)
        {
          jj -= local_cols;
          cell_col_offset += local_cols;
          local_cols = (int)A.get_n_columns_in_cell(0, ++cell_col);
        }

        bool t = false;
        const auto block = A.get_block(cell_row, cell_col, t);

        if (block != nullptr)
        {
          const int* rows = block->get_row_ptr();
          const int* columns = block->get_vector_columns();
          const double* entries = block->GetEntries();

          if (t)
          {
            int m0 = rows[jj];
            int m1 = rows[jj + 1];

            for (int k = m0; k < m1; k++)
            {
              if (columns[k] == ii)
              {
                temp_row[j] = entries[k];
              }
            }
          }
          else
          {
            int m0 = rows[ii];
            int m1 = rows[ii + 1];

            for (int k = m0; k < m1; k++)
            {
              if (columns[k] == jj)
              {
                temp_row[j] = entries[k];
              }
            }
          }
        }
      }

      M.setRow(i, temp_row.data());
      last_dof = dof[i];
    }

    return M;
  }

  template<>
  DenseMatrix DiagonalBlock<CompositeOperator<BlockVector>>(const CompositeOperator<BlockVector>&, const std::vector<int>&)
  {
    ErrThrow("CompositeOperator not supported");
    return DenseMatrix(0, 0);
  }
}

template<class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::buildBlocksGreedy()
{
  using namespace BlockJacobiStrategies;

  MatrixLookup<LinearOperator> A_(A);
  int n = (int)A_.n;
  const auto& diag_entries = A_.diag_entries;

  // sort DOF by diagonal entry in descending order
  std::vector<int> sorted_i(n);
  std::iota(sorted_i.begin(), sorted_i.end(), 0);
  std::stable_sort(sorted_i.begin(), sorted_i.end(), [&](int a, int b) -> bool
    {
      return diag_entries[a] > diag_entries[b];
    });

  // set of DOF already used in a block
  std::unordered_set<int> used_dof;

  // temp vectors for sparse row retrieval
  std::vector<int> next_neighbors;
  std::vector<double> next_neighbor_entries;

  // temp vectors for available neighbors at each DOF
  std::vector<int> available_neighbors;
  std::vector<double> available_neighbor_values;
  std::vector<int> neighbor_i(n);

  // collecting dof
  std::vector<int> temp_dof;

  // loop through all DOF
  for (int i = 0; i < n; i++)
  {
    // ordered by diagonal entry:
    // highest, lowest, 2nd highest, 2nd lowest...
    int i2 = i / 2;
    int ii = sorted_i[(i % 2 == 0) ? i2 : (n - 1 - i2)];

    if (used_dof.count(ii) > 0)
    {
      // this DOF is already in a block
      continue;
    }

    // mark used
    used_dof.insert(ii);

    // set first DOF for this block
    temp_dof.clear();
    temp_dof.push_back(ii);

    double d_i = diag_entries[ii];

    // get sparse row data
    A.get_row((unsigned)ii, next_neighbors, next_neighbor_entries);

    // check all neighbors for availability and compute coupling values
    available_neighbors.clear();
    available_neighbor_values.clear();
    for (int jj: next_neighbors)
    {
      if (used_dof.count(jj) == 0)
      {
        // not yet used in a block

        double va = A_.get(ii, jj);
        double vb = A_.get(jj, ii);
        double v = va * va + vb * vb;

        if (v > 0)
        {
          // neighbors are ordered by their coupling with the original DOF
          // scaled by diagonal similarity
          double d_j = diag_entries[jj];
          v = std::sqrt(v) * std::max(1e-6, std::min(d_j / d_i, d_i / d_j));

          available_neighbors.push_back(jj);
          available_neighbor_values.push_back(v);
        }
      }
    }

    // sort available neighbors by coupling in descending order
    neighbor_i.resize(available_neighbors.size());
    std::iota(neighbor_i.begin(), neighbor_i.end(), 0);
    std::stable_sort(neighbor_i.begin(), neighbor_i.end(), [&](int a, int b) -> bool
      {
        return available_neighbor_values[a] > available_neighbor_values[b];
      });

    int m = std::min((int)available_neighbors.size(), max_block_size - 1);

    for (int j = 0; j < m; j++)
    {
      int jj = available_neighbors[neighbor_i[j]];

      // mark used
      used_dof.insert(jj);

      // set next DOF for this block
      temp_dof.push_back(jj);
    }

    // add the block to D
    std::sort(temp_dof.begin(), temp_dof.end());
    auto M = DiagonalBlock(A, temp_dof);
    appendBlock(M, temp_dof);
  }
}

template<class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::buildBlocksLouvain()
{
  using namespace BlockJacobiStrategies;

  MatrixLookup<LinearOperator> A_(A);
  int n = (int)A_.n;

  // various temp structures to avoid reallocating all over the place
  std::vector<double> temp;
  std::set<int> temp_neighbors;
  std::unordered_set<int> temp_communities;
  std::vector<int> community_reindex;
  std::vector<std::vector<int>> temp_blocks;
  std::vector<std::vector<double>> temp_weights;
  std::vector<std::vector<int>> old_block_neighbors;

  // each DOF's immediate neighbors
  std::vector<std::vector<int>> neighbors(n);

  // the current blocks (each block is a list of DOFs)
  std::vector<std::vector<int>> current_blocks(n);

  // at each block, the sum of the attached block edges' weights
  std::vector<double> block_quality(n, 0.0);

  // index of the block each DOF is in
  std::vector<int> block_by_dof(n);

  // indices of each block's neighboring blocks
  std::vector<std::vector<int>> block_neighbors(n);

  // weights of the edges attached to each block, same order as block_neighbors
  std::vector<std::vector<double>> block_edge_weights(n);

  // the current communities (each community is a list of block indices)
  std::vector<std::vector<int>> communities(n);

  // index of the community each block is in
  std::vector<int> community_by_block(n);

  // total number of DOFs in each community
  std::vector<int> community_size(n);

  // twice the sum of all block edge weights
  double total_quality = 0.0;

  Output::info<4>("Block-Jacobi", "Initializing 1-blocks and communities...");

  // initial state:
  // each DOF gets its own block, each block gets its own community
  for (int r = 0; r < n; r++)
  {
    communities[r] = { r };
    community_size[r] = 1;
    community_by_block[r] = r;

    current_blocks[r] = { r };
    block_by_dof[r] = r;
    A.get_row((unsigned)r, neighbors[r], temp);
    block_neighbors[r] = neighbors[r];

    block_edge_weights[r].resize(neighbors[r].size());
    for (unsigned i = 0; i < neighbors[r].size(); i++)
    {
      // the initial weight between two 1-DOF blocks is the coupling of those
      // two DOFs, with zero diagonal (see MatrixLookup::get_weight)

      int c = neighbors[r][i];
      double v = A_.get_weight(r, c);
      block_edge_weights[r][i] = v;
      block_quality[r] += v;
    }

    total_quality += block_quality[r];
  }

  Output::info<4>("Block-Jacobi", "Total quality = ", total_quality);

  // weight lookup helper
  auto block_edge_weight = [&] (int i, int j) -> double
  {
    int m = utilities::binarySearch(block_neighbors[i], j);
    return m < 0 ? 0.0 : block_edge_weights[i][m];
  };

  // this flag is set in the inner loop if anything changes
  bool communities_changed = false;
  bool final_stage = false;

  do
  {
    // new iteration of the outer loop, reset the change flag
    communities_changed = false;

    Output::info<4>("Block-Jacobi", "Building communities...");

    // compute initial modularity
    double modularity = 0.0;
    for (const auto& community: communities)
    {
      for (int a: community)
      {
        for (int b: community)
        {
          modularity += block_edge_weight(a, b) - (block_quality[a] * block_quality[b]) / total_quality;
        }
      }
    }
    modularity /= total_quality;

    Output::info<4>("Block-Jacobi", "Initial modularity: ", modularity);

    int num_blocks = (int)current_blocks.size();

    // counts the number of blocks moved in each iteration of the inner loop
    int blocks_moved = 0;

    do
    {
      // new iteration of the inner loop, reset move counter
      blocks_moved = 0;

      double impr_min = 1e+12;
      double impr_max = 0.0;

      // loop over all blocks and see if moving each makes sense
      for (int i = 0; i < num_blocks; i++)
      {
        // initial "best" value is zero since we're only interested in positive
        // changes to the modularity
        double best_mod_improvement = threshold * total_quality;
        int best_community = -1;
        int block_size = (int)current_blocks[i].size();

        // make sure we only check each other community we might move the block to
        temp_communities.clear();
        temp_communities.insert(community_by_block[i]);

        const auto& current_community = communities[community_by_block[i]];

        // compute the block's contribution to the modularity within its
        // current community (actually only half the off-diagonal terms, since
        // the diagonal term does not change and the weights are symmetric)
        double mod_loss = 0.0;
        for (int a: current_community)
        {
          mod_loss += block_edge_weight(a, i) - (block_quality[a] * block_quality[i]) / total_quality;
        }

        // check each neighboring block's community
        for (int j: block_neighbors[i])
        {
          int comm_j = community_by_block[j];
          if (temp_communities.count(comm_j) == 0)
          {
            // we haven't looked at this community yet
            temp_communities.insert(comm_j);

            if (community_size[comm_j] + block_size > max_block_size)
            {
              // can't move this block to this community, it'd be too big
              continue;
            }
            else
            {
              const auto& proposed_community = communities[comm_j];

              // compute the block's contribution to the modularity if moved to
              // the proposed community (actually only half the off-diagonal
              // terms, since the diagonal term does not change and the weights
              // are symmetric)
              double mod_gain = 0.0;
              for (int a: proposed_community)
              {
                mod_gain += block_edge_weight(a, i) - (block_quality[a] * block_quality[i]) / total_quality;
              }
              mod_gain -= mod_loss;

              if (mod_gain > best_mod_improvement)
              {
                // this is better, update the best value and index
                best_mod_improvement = mod_gain;
                best_community = comm_j;
              }
            }
          }
        }

        if (best_community >= 0)
        {
          // found a better community for this block, move it

          auto& old_community = communities[community_by_block[i]];
          auto& new_community = communities[best_community];

          community_size[community_by_block[i]] -= block_size;
          community_size[best_community] += block_size;

          int m = utilities::binarySearch(old_community, i);
          if (m >= 0)
          {
            // erase in-place to maintain order
            old_community.erase(old_community.begin() + m);
          }

          // insert in correct position to maintain order
          int ncs = (int)new_community.size();
          int s = 0;
          for (; s < ncs; s++)
          {
            if (new_community[s] > i)
            {
              break;
            }
          }
          new_community.insert(new_community.begin() + s, i);

          community_by_block[i] = best_community;

          ++blocks_moved;

          // update delta-mod range
          best_mod_improvement /= total_quality;
          impr_min = std::min(impr_min, best_mod_improvement);
          impr_max = std::max(impr_max, best_mod_improvement);
        }
      }

      if (blocks_moved > 0)
      {
        // we moved some blocks! set the flag and check for improvement
        communities_changed = true;

        // compute the new modularity
        double new_modularity = 0.0;
        for (const auto& community: communities)
        {
          for (int a: community)
          {
            for (int b: community)
            {
              new_modularity += block_edge_weight(a, b) - (block_quality[a] * block_quality[b]) / total_quality;
            }
          }
        }
        new_modularity /= total_quality;

        Output::info<4>("Block-Jacobi", "Moved ", blocks_moved, " blocks, estimated delta mod range:",
          "\n", impr_min, " to ", impr_max, ". New modularity: ", new_modularity);

        if (new_modularity <= modularity)
        {
          // this will sometimes happen due to noise in the delta-mod
          // computations. break here to avoid infinite cycling.
          Output::info<4>("Block-Jacobi", "Modularity did not improve.");
          break;
        }

        modularity = new_modularity;
      }
      else
      {
        // done!
        Output::info<4>("Block-Jacobi", "Moved no more blocks.");
      }
    }
    while (blocks_moved > 0);

    if (communities_changed)
    {
      // Something changed. We have to update the blocks.

      Output::info<4>("Block-Jacobi", "Merging blocks and updating adjacencies...");

      // Form one new block per nonempty community by merging its blocks.
      // The new blocks are written into temp_blocks, which is empty here.
      community_reindex.clear();
      int community_count = 0;
      for (const auto& community: communities)
      {
        if (community.size() > 0)
        {
          // this one isn't empty, merge its blocks

          temp_blocks.emplace_back(0);
          auto& new_block = temp_blocks.back();
          int new_block_index = (int)temp_blocks.size() - 1;

          // count up the size of the new block and reserve
          int new_block_size = 0;
          for (int i: community)
          {
            new_block_size += (int)current_blocks[i].size();
          }
          new_block.reserve(new_block_size);

          if (new_block_size != community_size[community_count])
          {
            ErrThrow("Community size mismatch encountered: ", new_block_size, " vs. ", community_size[community_count]);
          }

          if (new_block_size > max_block_size)
          {
            ErrThrow("Block of size ", new_block_size, " encountered");
          }

          for (int i: community)
          {
            // add the DOFs from old block i to the new block
            for (int j: current_blocks[i])
            {
              new_block.push_back(j);
              block_by_dof[j] = new_block_index;
            }
          }

          // make sure the new block is sorted
          std::sort(new_block.begin(), new_block.end());

          // this is the new block's index as a community
          // in the old community list
          community_reindex.push_back(community_count);
        }

        community_count++;
      }

      // Swap block lists and discard the old blocks.
      std::swap(current_blocks, temp_blocks);
      temp_blocks.clear();

      int num_new_blocks = current_blocks.size();

      Output::info<4>("Block-Jacobi", "Built ", num_new_blocks, " new blocks.");

      // Update block adjacencies.

      // we need the old neighborhood information to efficiently update the
      // edge weights. otherwise we'd have to laboriously loop over a bunch of
      // DOF pairs
      std::swap(block_neighbors, old_block_neighbors);

      // set up the new arrays
      block_neighbors.resize(num_new_blocks);
      block_quality.resize(num_new_blocks);
      temp_weights.resize(num_new_blocks);
      total_quality = 0.0;
      for (int i = 0; i < num_new_blocks; i++)
      {
        // gather the new neighborhood information into a set
        temp_neighbors.clear();
        for (int j: current_blocks[i])
        {
          for (int k: neighbors[j])
          {
            temp_neighbors.insert(block_by_dof[k]);
          }
        }

        // copy the new neighborhood data. note that temp_neighbors is an
        // ordered set, so these are in order
        block_neighbors[i].assign(temp_neighbors.begin(), temp_neighbors.end());

        int n_neighbors = (int)block_neighbors[i].size();

        // update the block edge weights by adding up all the
        // corresponding old block edge weights
        block_quality[i] = 0.0;
        temp_weights[i].resize(n_neighbors);
        for (int j = 0; j < n_neighbors; j++)
        {
          if (block_neighbors[i][j] != i)
          {
            // index of the old community corresponding to
            // this new neighboring block
            int jj = community_reindex[block_neighbors[i][j]];

            // loop over all the old blocks in the old community corresponding
            // to this new block
            double v = 0.0;
            for (int old_block: communities[community_reindex[i]])
            {
              // loop over the old neighboring blocks
              const auto& old_neighbors = old_block_neighbors[old_block];
              int n = old_neighbors.size();
              for (int k = 0; k < n; k++)
              {
                // add the block edge's weight if the old neighboring block was
                // in the old community corresponding to this new neighbor block
                if (community_by_block[old_neighbors[k]] == jj)
                {
                  v += block_edge_weights[old_block][k];
                }
              }
            }

            // set weight
            temp_weights[i][j] = v;
            block_quality[i] += v;
          }
          else
          {
            // zero diagonal weights
            temp_weights[i][j] = 0.0;
          }
        }

        total_quality += block_quality[i];
      }

      // swap the weight lists and discard the old
      // weights and neighborhood information
      std::swap(block_edge_weights, temp_weights);
      temp_weights.clear();
      old_block_neighbors.clear();

      Output::info<4>("Block-Jacobi", "Total quality = ", total_quality);

      // reset communities (each new block in a new community of its own)
      communities.resize(num_new_blocks);
      for (int i = 0; i < num_new_blocks; i++)
      {
        communities[i] = { i };
        community_size[i] = (int)current_blocks[i].size();
        community_by_block[i] = i;
      }
    }
    else if (final_stage)
    {
      break;
    }
    else
    {
      // Final local stage: convert current blocks into communities of
      // single-DOF blocks

      Output::info<4>("Block-Jacobi", "Preparing final stage...");

      for (auto& c: communities)
      {
        if (c.size() != 1)
        {
          ErrThrow("Final stage should not be reached with empty or multi-block communities ", c.size());
        }
      }

      int num_blocks = current_blocks.size();

      communities.resize(num_blocks);
      community_size.resize(num_blocks);
      community_by_block.resize(n);
      for (int i = 0; i < num_blocks; i++)
      {
        communities[i] = current_blocks[i];

        community_size[i] = (int)communities[i].size();
        for (int j: communities[i])
        {
          community_by_block[j] = i;
        }

        if (community_size[i] == 0)
        {
          ErrThrow("Final stage should not be reached with empty blocks");
        }

        if (community_size[i] > max_block_size)
        {
          ErrThrow("Final stage should not be reached with oversized blocks");
        }
      }

      current_blocks.resize(n);
      block_by_dof.resize(n);
      block_neighbors.resize(n);
      block_edge_weights.resize(n);
      block_quality.resize(n);
      total_quality = 0.0;
      for (int r = 0; r < n; r++)
      {
        current_blocks[r] = { r };
        block_by_dof[r] = r;
        block_neighbors[r] = neighbors[r];

        block_edge_weights[r].resize(neighbors[r].size());
        for (unsigned i = 0; i < neighbors[r].size(); i++)
        {
          int c = neighbors[r][i];
          double v = A_.get_weight(r, c);
          block_edge_weights[r][i] = v;
          block_quality[r] += v;
        }

        total_quality += block_quality[r];
      }

      Output::info<4>("Block-Jacobi", "Built ", communities.size(), " communities, ",
        current_blocks.size(), " blocks. Total quality = ", total_quality);

      final_stage = true;
    }
  }
  while (communities_changed || final_stage);

  // done - build block diagonal matrix
  // std::vector<double> temp_row(max_block_size);
  for (const auto& b: current_blocks)
  {
    int m = (int)b.size();

    if (m == 0 || m > max_block_size)
    {
      // this is impossible unless a terrible bug happens above
      ErrThrow("Louvain algorithm created block of size ", m);
    }
    else
    {
      // add the block to D
      auto M = DiagonalBlock<LinearOperator>(A, b);
      appendBlock(M, b);
    }
  }
}

template <class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::appendBlock(DenseMatrix &M, const std::vector<int>& dof)
{
  int m = M.getNRows();

  // perform LU decomposition and finish block setup
  try
  {
    switch (m)
    {
      case 1:
        {
          double d = M.getEntry(0, 0);
          if (d == 0.0)
          {
            ErrThrow("Found a 1x1 block corresponding to a zero on the diagonal!");
          }
          else
          {
            diagonal_dof.push_back(dof[0]);
            diagonal_values.push_back(d);
            return;
          }
        }
        break;
      case 2:
        {
          double a00 = M.getEntry(0, 0);
          double a01 = M.getEntry(0, 1);
          double a10 = M.getEntry(1, 0);
          double a11 = M.getEntry(1, 1);

          double det = a00 * a11 - a01 * a10;
          if (std::abs(det) < 1e-32)
          {
            ErrThrow("Found a (nearly) singular 2x2 block!");
          }
          else
          {
            double b00 =  a11 / det;
            double b01 = -a01 / det;
            double b10 = -a10 / det;
            double b11 =  a00 / det;

            dof_2x2.push_back(dof[0]);
            dof_2x2.push_back(dof[1]);
            invM_2x2.push_back(b00);
            invM_2x2.push_back(b01);
            invM_2x2.push_back(b10);
            invM_2x2.push_back(b11);

            return;
          }
        }
        break;
      default:
        M.decomposeLU();
        break;
    }

    blocks.emplace_back(M, dof);
  }
  catch (...)
  {
    if (m > 1)
    {
      M.print("M");
      Output::warn("Block-Jacobi", "Block matrix may be singular, reverting to 1x1 blocks");

      for (int i = 0; i < m; i++)
      {
        DenseMatrix N(1, 1);
        N.setEntry(0, 0, M.getEntry(i, i));
        appendBlock(N, { dof[i] });
      }
    }
    else
    {
      throw;
    }
  }
}

template <class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::update()
{
  // no blocks
  blocks.clear();

  diagonal_dof.clear();
  diagonal_values.clear();

  dof_2x2.clear();
  invM_2x2.clear();

  unsigned int v = Output::getVerbosity();

  if (strategy == "greedy")
  {
    buildBlocksGreedy();
  }
  else if (strategy == "louvain")
  {
    buildBlocksLouvain();
  }
  else
  {
    ErrThrow("Block Jacobi strategy '", strategy, "' not supported.");
  }

  if (v >= 5)
  {
    // output all blocks

    for (const Block& b: blocks)
    {
      std::ostringstream os;

      os << setprecision(4) << std::fixed;

      os << "\n";
      for (int j = 0; j < b.n; j++)
      {
        os << "\t\t" << b.dof[j];
      }

      for (int j = 0; j < b.n; j++)
      {
        os << "\n";
        for (int k = 0; k < b.n; k++)
        {
          os << "\t\t" << b.M.getEntry(j, k);
        }
      }

      Output::info("Block-Jacobi", "Block of size ", b.n, ":\n", os.str());
    }
  }

  if (v >= 3)
  {
    // count blocks by size

    unsigned int n = A.get_n_total_rows();

    std::vector<int> block_size_count(max_block_size, 0);
    for (const Block& b: blocks)
    {
      block_size_count[b.n - 1]++;
    }

    block_size_count[0] += (int)diagonal_dof.size();
    block_size_count[1] += (int)dof_2x2.size() / 2;

    int n_blocks = (int)(blocks.size() + diagonal_dof.size() + dof_2x2.size() / 2);

    if (v >= 4)
    {
      for (int i = 0; i < max_block_size; i++)
      {
        if (block_size_count[i] > 0)
        {
          Output::info("Block-Jacobi",
            block_size_count[i], " blocks of size ", i + 1);
        }
      }
    }

    Output::info("Block-Jacobi", "Built ", n_blocks, " blocks for ", n,
      " rows, avg block size ", double(n) / double(n_blocks), ".");
  }

  Output::setVerbosity(v);
}

template <class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::apply(const Vector &z, Vector &r) const
{
  std::vector<double> scratch(max_block_size);

  for (const Block& b: blocks)
  {
    b.apply(z, r, scratch.data());
  }

  for (unsigned int i = 0; i < diagonal_dof.size(); i++)
  {
    r[diagonal_dof[i]] = z[diagonal_dof[i]] / diagonal_values[i];
  }

  unsigned int n_2x2 = dof_2x2.size() / 2;
  for (unsigned int i = 0; i < n_2x2; i++)
  {
    int d0 = dof_2x2[2 * i + 0];
    int d1 = dof_2x2[2 * i + 1];

    double b00 = invM_2x2[4 * i + 0];
    double b01 = invM_2x2[4 * i + 1];
    double b10 = invM_2x2[4 * i + 2];
    double b11 = invM_2x2[4 * i + 3];

    double z0 = z[d0];
    double z1 = z[d1];

    r[d0] = b00 * z0 + b01 * z1;
    r[d1] = b10 * z0 + b11 * z1;
  }
}

template <class LinearOperator, class Vector>
Preconditioner_block_jacobi<LinearOperator, Vector>::Block::Block(const DenseMatrix &M, const std::vector<int>& dof)
: M(std::move(M)), n((int)dof.size()), dof(dof)
{
  if (n < 3)
  {
    ErrThrow("Matrix is of size ", n, " - this should not occur or be handled as a special case");
  }

  if (n != this->M.getNRows() || n != this->M.getNColumns())
  {
    ErrThrow("Matrix is not square or does not match DOF vector length");
  }

  if (!this->M.getEntriesLU() || !this->M.getPivotsLU())
  {
    ErrThrow("Matrix is not set up for solving");
  }
}

template <class LinearOperator, class Vector>
void Preconditioner_block_jacobi<LinearOperator, Vector>::Block::apply(const Vector& z, Vector& r,
  double* scratch) const
{
  for (int i = 0; i < n; i++)
  {
    scratch[i] = z[dof[i]];
  }

  M.solve(scratch);

  for (int i = 0; i < n; i++)
  {
    r[dof[i]] = scratch[i];
  }
}

template class Preconditioner_block_jacobi<BlockFEMatrix, BlockVector>;
template class Preconditioner_block_jacobi<CompositeOperator<BlockVector>, BlockVector>;

#ifndef _MPI
template class Preconditioner_block_jacobi<BlockMatrix, BlockVector>;
#endif