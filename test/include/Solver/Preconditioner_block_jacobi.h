#ifndef __PRECONDITIONER_BLOCK_JACOBI__
#define __PRECONDITIONER_BLOCK_JACOBI__

#include <IterativeMethod.h>
#include <DenseMatrix.h>

/// @brief Block-Jacobi preconditioner. Decomposes A = D + N where D is
///        block-diagonal and N is zero within the blocks of D, then
///        preconditions by solving Dr = z directly.
///
///        The following block selection strategies are available:
///          - "greedy": A simple and fast greedy method.
///          - "louvain": A modified Louvain method.
///
///        Greedy method (strategy == "greedy"):
///          - In the below, an "available" DOF is one not yet used in a block.
///          - Sort DOF by diagonal, alternating between highest and lowest.
///          - In that order, consider each available DOF i.
///          - Take all available immediate neighbors of i ordered by their
///            coupling with i until there are no more neighbors or the maximum
///            block size is reached.
///          - Append the corresponding block to D.
///
///        Modified Louvain method (strategy == "louvain"):
///          - Consider the DOF graph with edges weighted by coupling.
///          - We will be heuristically optimizing this graph's "modularity" Q
///            when partitioned into "communities" c. The term "community" is
///            due to the method's origin in the study of social networks.
///
///            If a_ij are the edge weights, let
///              k_i = \sum_j a_ij
///            be the sum of the weights of the edges at vertex i and let
///              K = \sum_i k_i
///            be twice the sum of all edge weights. Then the modularity is
///
///              Q = 1/K \sum_c \sum_{i,j \in c} (a_ij - k_i k_j / K).
///
///            The modularity is a measure of the proportion of "in-community"
///            edge mass.
///
///            A significant difference from many expositions of this method
///            is that diagonal weights a_ii are always zero, even for the
///            fused block graphs in later steps. This is because over-merging
///            is a major issue in community detection when studying a social
///            networks, but is prevented by the block size limit in our case.
///
///          - Initialize a list of communities with each vertex of the graph
///            in a single-member community of its own.
///          - Loop through each vertex of the graph and move it to the
///            neighboring community such that the modularity is improved
///            as much as possible and the new community's total number of DOFs
///            does not exceed max_block_size. (Do not move vertices where no
///            such positive improvement of Q is possible.) Repeat this step
///            until no more vertices are moved.
///          - Merge each community into a block and apply the above procedure
///            to the new block graph with edges weighted by the sum of
///            inter-community edges. Repeat until constant.
///          - Build D from the final blocks.
template <class LinearOperator, class Vector>
class Preconditioner_block_jacobi : public Preconditioner<Vector>
{
  public:
    explicit Preconditioner_block_jacobi(const LinearOperator& A,
      std::string strategy, double threshold,
      int max_block_size);
    virtual ~Preconditioner_block_jacobi() = default;

    /// @brief Solves Dr = z.
    void apply(const Vector& z, Vector& r) const override final;

    void update() override final;

  private:

    /// @brief Performs greedy block selection.
    void buildBlocksGreedy();

    /// @brief Performs Louvain block selection.
    void buildBlocksLouvain();

    void appendBlock(DenseMatrix &M, const std::vector<int>& dof);

    /// @brief Internal structure representing a block on the diagonal of D
    ///        together with information on which DOF it represents
    struct Block
    {
    public:
      Block(const DenseMatrix &M, const std::vector<int>& dof);

      /// @brief Solves Mr_I = z_I.
      void apply(const Vector& z, Vector& r, double* scratch) const;

      /// @brief Block matrix
      const DenseMatrix M;

      /// @brief Block size
      const int n;

      /// @brief DOF index set I affected by this block
      const std::vector<int> dof;
    };

    const LinearOperator& A;
    const std::string strategy;
    const double threshold;
    const int max_block_size;

    // @brief Representation of D by a series of blocks
    std::vector<Block> blocks;

    std::vector<int> diagonal_dof;
    std::vector<double> diagonal_values;

    std::vector<int> dof_2x2;
    std::vector<double> invM_2x2;
};

#endif // __PRECONDITIONER_BLOCK_JACOBI__
