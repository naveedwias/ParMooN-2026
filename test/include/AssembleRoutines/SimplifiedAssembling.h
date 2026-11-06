#ifndef SIMPLIFIED_ASSEMBLING_H
#define SIMPLIFIED_ASSEMBLING_H

#include <ParameterDatabase.h>
#include <QuadFormula.h>
#include <Collection.h>
#include <FiniteElement.h>
#include <BaseFunctions.h>
#include <MultiVector.h>

#include <Enumerations_fe.h>
#include <templateNames.h>

#include <memory>
#include <vector>
#include <array>
#include <functional>

class FEMatrix;
class BlockFEMatrix;
class BlockVector;

/// @brief Structure containing data necessary for matrix
///        assembly at a given point
struct SimplifiedAssemblingData
{
public:
  SimplifiedAssemblingData(double** matrix, double* rhs,
    int n_rows, int n_cols,
    const MultiVector<double, 2>& row_basis,
    const MultiVector<double, 2>& col_basis,
    const std::vector<MultiVector<double, 2>>& fe_func,
    const int& cell_index, const double* point,
    const int block_row, const int block_col);

  /// @brief Output matrix (n_cols column pointers)
  double** matrix;

  /// @brief Output right-hand side vector (column pointer)
  double* rhs;

  /// @brief Number of rows in the system
  const int n_rows;

  /// @brief Number of columns in the system
  const int n_cols;

  /// @brief Row space basis values and derivatives in requested order.
  ///        if L_j are the requested derivatives and v^r is the r-th row basis
  ///        function for this block, then:
  ///          row_basis[{j, r}] = L_j(v^r).
  const MultiVector<double, 2>& row_basis;

  /// @brief Column space basis values and derivatives in requested order.
  ///        if L_j are the requested derivatives and v^r is the r-th column
  ///        basis function for this block, then:
  ///          col_basis[{j, r}] = L_j(v^r).
  const MultiVector<double, 2>& col_basis;

  /// @brief Input FE function values and derivatives in requested order.
  ///        if L_j are the requested derivatives and f_i^r is the i-th
  ///        component of the r-th input FE function, then:
  ///          fe_func[r][{i, j}] = L_j(f_i^r).
  const std::vector<MultiVector<double, 2>>& fe_func;

  const int& cell_index;
  const double* point;
  const int block_row;
  const int block_col;
};

/// @brief Function type for simplified assembly. Called at each quadrature point.
///
/// Signature:
///
/// void AssembleLocal(double weight, double hK,
///   const SimplifiedAssemblingData& data);
///
/// When called:
///   * weight is the quadrature weight associated with the current point
///   * hK is a measure of the local cell's size
///   * data contains various necessary assembly buffers.
typedef std::function<void(double, double, const SimplifiedAssemblingData&)> AssembleLocalSimplified;

/// @brief Generic simplified assembler class.
template<int d>
class SimplifiedAssembling
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a simplified assembler with possibly distinct test and
    ///        ansatz spaces.
    SimplifiedAssembling(std::vector<std::shared_ptr<const FESpace>> test_spaces,
      std::vector<std::shared_ptr<const FESpace>> ansatz_spaces,
      MultiIndex_vector test_derivatives,
      MultiIndex_vector ansatz_derivatives);

    /// @brief Construct a simplified assembler with identical test and ansatz spaces.
    SimplifiedAssembling(std::vector<std::shared_ptr<const FESpace>> spaces,
      MultiIndex_vector derivatives);

    /// @brief Add an assembly function with scale information.
    void AddAssembly(AssembleLocalSimplified assembly, int r, int c, double m = 1.0);

    /// @brief Add an FE function.
    void AddFEFunction(const double* function,
      const MultiIndex_vector& derivatives, bool is_test = false);

    /// @brief Clear assembly data.
    void Reset();

    /// @brief Perform the actual assembly.
    void Assemble(double multiplier,
      BlockFEMatrix *matrix,
      BlockVector *rhs,
      bool clear = false);

    /// @brief Perform the actual assembly.
    void Assemble(double multiplier,
      FEMatrix *matrix,
      double *rhs,
      bool clear = false);

    /// @brief Output-less assembly. Use for assembly functions that write to
    ///        external variables, e.g. for error integrals.
    void Assemble(double multiplier);

    /// @brief Shortcut for mass matrix assembly.
    void AssembleMass(double multiplier,
      BlockFEMatrix *matrix,
      bool clear = false);

    /// @brief Shortcut for stiffness matrix assembly.
    void AssembleStiffness(double multiplier,
      BlockFEMatrix *matrix,
      bool clear = false,
      bool scaled = false);

    /// @brief Assemble M_ij = (v_i, v_j)
    static std::shared_ptr<FEMatrix> AssembleMass(std::shared_ptr<const FESpace> space, double m = 1.0);

    /// @brief Assemble A_ij = (\nabla v_i, \nabla v_j)
    static std::shared_ptr<FEMatrix> AssembleStiffness(std::shared_ptr<const FESpace> space, double m = 1.0, bool scaled = false);

    /// @brief For ansatz space V and test space Q, assemble B_ij = (v_j, q_i)
    static std::shared_ptr<FEMatrix> AssembleL2Product(
      std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
      double m = 1.0);

    /// @brief For ansatz space V and test space Q, assemble B_ij = (\nabla v_j, \nabla q_i)
    static std::shared_ptr<FEMatrix> AssembleH1Product(
      std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
      double m = 1.0);

    /// @brief For ansatz space V^d and test space Q, assemble D_ij,k = (div(v_j e_k), q_i)
    static std::vector<std::shared_ptr<FEMatrix>> AssembleDivProduct(
      std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
      double m = 1.0);

    /// @brief For ansatz space V^d and test space Q, assemble D_ij,k = (v_j e_k, \nabla q_i)
    static std::vector<std::shared_ptr<FEMatrix>> AssembleGradProduct(
      std::shared_ptr<const FESpace> test_space, std::shared_ptr<const FESpace> ansatz_space,
      double m = 1.0);

  protected:
    /// @brief Internal struct storing information on an assembly function.
    struct AssemblyInfo
    {
      public:
        AssemblyInfo(AssembleLocalSimplified func, int row, int column, float multiplier);

        const AssembleLocalSimplified func;
        const int row;
        const int column;
        const float multiplier;
    };

    /// @brief Internal struct storing information on an FE function.
    struct FunctionInfo
    {
      public:
        FunctionInfo(const double* func,
          const MultiIndex_vector& derivatives,
          const bool is_test);

        /// @brief Nodal/modal vector.
        const double* func;

        /// @brief Which derivatives do we need?
        const MultiIndex_vector derivatives;

        /// @brief Determines whether this function lives in the test or ansatz
        ///        space.
        const bool is_test;
    };

    struct AssemblyLoop
    {
      public:
        AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, BlockFEMatrix* mat, BlockVector *rhs, bool clear);
        AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, FEMatrix* mat, double *rhs, bool clear);
        AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier, bool block_type, bool has_matrix, bool has_rhs);
        AssemblyLoop(SimplifiedAssembling<d> &assembler, double multiplier);
        ~AssemblyLoop();

        void Assemble();

        void TransferBlockData(const AssemblyInfo &info, int c);
        void TransferFlatData(const AssemblyInfo &info, int c);

      protected:
        SimplifiedAssembling<d> &assembler;
        const double multiplier;
        const bool block_type;

        const bool has_matrix;
        const bool has_rhs;

        BlockFEMatrix* block_fe_mat;
        FEMatrix* fe_mat;

        BlockVector* block_rhs;
        double* flat_rhs;

        bool distinct_bases;
        bool* sd;

        unsigned int max_row_dim;
        unsigned int max_col_dim;

        std::vector<double> local_matrix_data;
        std::vector<double*> local_matrix;
        std::vector<double> local_rhs;

        std::vector<int> block_offset;

        int cell_index;
        std::array<double, d> point;

        std::vector<SimplifiedAssemblingData> data;
    };

    /// @brief Are test and ansatz spaces and bases distinct?
    bool DistinctBases();

    /// @brief Helper function to set up the derivative lookup for a given
    ///        set of requested and retrieved multi-indices, to be called only
    ///        from SetDerivativeLookups().
    void SetDerivativeLookup(const MultiIndex_vector& derivatives,
      const MultiIndex_vector& indices, std::vector<int>& lookup);

    /// @brief Set up the derivative lookup structures for the bases.
    void SetDerivativeLookups();

    /// @brief Set up the buffers and lookup pointers for the bases.
    void SetBasisAndFunctionStorage();

    /// @brief Assemble the test basis at cell c, quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalTestBasis(int c, int p);

    /// @brief Assemble the ansatz basis at cell c, quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalAnsatzBasis(int c, int p);

    /// @brief Assemble all FE functions at cell c,
    ///        quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalFEFunctions(int c, int p);

    /// @brief Helper function to retrieve the FiniteElement and BaseFunctions
    ///        for a given multi-component FE space on cell c.
    ///        To be called only from RetrieveLocalFEData().
    void RetrieveLocalElements(const std::vector<std::shared_ptr<const FESpace>>& spaces,
      int c, unsigned int offset);

    /// @brief Helper function to retrieve the local basis values for a given
    ///        multi-component FE space on the current cell.
    ///        To be called only from RetrieveLocalFEData().
    void RetrieveLocalBasis(const std::vector<std::shared_ptr<const FESpace>>& spaces,
      const MultiIndex_vector& indices, unsigned int offset);

    /// @brief Retrieve bases and quadrature data for all FE spaces on cell c.
    ///        To be called only from the cell loop of Assemble(). Assumes
    ///        storage arrays and derivative lookups are in the correct state.
    TQuadFormula RetrieveLocalFEData(const TCollection* collection,
      int c, bool* sd);

    /// @brief Test space. May have multiple components.
    const std::vector<std::shared_ptr<const FESpace>> test_spaces;

    /// @brief Ansatz space. May have multiple components.
    const std::vector<std::shared_ptr<const FESpace>> ansatz_spaces;

    /// @brief Number of components in the test space.
    const int test_vector_dim;

    /// @brief Number of components in the ansatz space.
    const int ansatz_vector_dim;

    /// @brief Which derivatives of the test basis do we need?
    const MultiIndex_vector test_derivatives;

    /// @brief Which derivatives of the ansatz basis do we need?
    const MultiIndex_vector ansatz_derivatives;

    /// @brief Assembly functions to call.
    std::vector<AssemblyInfo> assembly_functions;

    /// @brief FE functions.
    std::vector<FunctionInfo> fe_functions;

    /// @brief FE function values, indexed as [function index][vector dimension, functional]
    std::vector<MultiVector<double, 2>> fe_values;

    /// @brief Basis function values, indexed as [vector dimension] [functional, basis function]
    std::vector<MultiVector<double, 2>> test_values, ansatz_values;

    /// @brief Do we need the test FE space's second derivatives?
    bool test_sd;

    /// @brief Do we need the ansatz FE space's second derivatives?
    bool ansatz_sd;

    /// @brief Derivatives of the test/ansatz basis to retrieve.
    MultiIndex_vector test_indices, ansatz_indices;

    /// @brief Mapping of the requested derivatives to the retrieved for the
    ///        basis functions.
    std::vector<int> test_derivative_lookup, ansatz_derivative_lookup;

    /// @brief Mapping of the requested derivatives to the retrieved
    ///        for the FE functions.
    std::vector<std::vector<int>> fe_derivative_lookup;

    /// @brief Buffer to store all full-order basis values for the current
    ///        cell during assembly.
    std::vector<std::vector<double**>> orig_basis_values;

    /// @brief Buffer to store all spaces' FiniteElement for the current cell
    ///        during assembly.
    std::vector<const FiniteElement*> elements;

    /// @brief Buffer to store all spaces' BaseFunctions for the current cell
    ///        during assembly.
    std::vector<const BaseFunctions*> bf;

    bool setup_finished;
};

#endif