#ifndef __PRECONDITIONER_SPARSE_APPROXIMATE_INVERSE__
#define __PRECONDITIONER_SPARSE_APPROXIMATE_INVERSE__

#include <IterativeMethod.h>
#include <memory>

/// @brief Sparse approximate inverse preconditioner. Approximates A^{-1} by
///        computing a matrix M with the same sparsity pattern as A such that
///        the Frobenius norm of I - AM is minimal.
///        For BlockFEMatrix with symmetric sparsity pattern only.
template <class LinearOperator, class Vector>
class Preconditioner_sparse_approximate_inverse : public Preconditioner<Vector>
{
  public:
    /// @brief Construct an SAI preconditioner for a given matrix A.
    ///        If symmetric is true, A is assumed to be symmetric.
    explicit Preconditioner_sparse_approximate_inverse(const LinearOperator& A,
      bool symmetric = false, bool active_rows = false);
    virtual ~Preconditioner_sparse_approximate_inverse() = default;

    /// @brief Computes r = Mz.
    void apply(const Vector& z, Vector& r) const override final;

    void update() override final;

  private:
    const LinearOperator& A;
    const bool symmetric;
    const bool active_rows;
    std::shared_ptr<LinearOperator> M;
};

#endif // __PRECONDITIONER_BLOCK_JACOBI__
