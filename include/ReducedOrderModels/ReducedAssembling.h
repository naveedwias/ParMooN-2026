#ifndef REDUCED_ASSEMBLING_H
#define REDUCED_ASSEMBLING_H

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

/// @brief Structure containing data necessary for matrix
///        assembly at a given point
struct ReducedAssemblingData
{
public:
  /// @brief Output matrix (n_cols column pointers)
  double** matrix;

  /// @brief Output right-hand side vector (column pointer)
  double* rhs;

  /// @brief Output values
  double* val;

  /// @brief Number of rows in the system
  const int n_rows;

  /// @brief Number of columns in the system
  const int n_cols;

  /// @brief Row space basis value and derivatives in requested order.
  ///        if L_j are the requested derivatives and v^r_i are the components of
  ///        the r-th basis function, then:
  ///          row_basis[{i, j, r}] = L_j(v^r_i).
  const MultiVector<double, 3>& row_basis;

  /// @brief Column space basis value and derivatives in requested order.
  ///        if L_j are the requested derivatives and v^r_i are the components of
  ///        the r-th basis function, then:
  ///          col_basis[{i, j, r}] = L_j(v^r_i).
  const MultiVector<double, 3>& col_basis;

  /// @brief Reduced FE function value and derivatives in requested order.
  ///        if L_j are the requested derivatives and u^r_i are the components
  ///        of the r-th reduced FE function, then:
  ///          reduced_func[r][{i, j}] = L_j(u^r_i).
  ///
  ///        The functions are given in the order in which they were added to
  ///        the assembler.
  const std::vector<MultiVector<double, 2>>& reduced_func;

  /// @brief Full-order FE function value and derivatives in requested order.
  ///        if L_j are the requested derivatives and u^r_i are the components
  ///        of the r-th full-order FE function, then:
  ///          full_func[r][{i, j}] = L_j(u^r_i).
  ///
  ///        The functions are given in the order in which they were added to
  ///        the assembler.
  const std::vector<MultiVector<double, 2>>& full_func;

  /// @brief Auxiliary FE function value and derivatives in requested order.
  ///        if L_j are the requested derivatives and u^r_i are the components
  ///        of the r-th aux FE function, then:
  ///          aux_func[r][{i, j}] = L_j(u^r_i).
  ///
  ///        The functions are given in the order of the aux spaces used,
  ///        *then* in the order in which they were added to the assembler.
  const std::vector<MultiVector<double, 2>>& aux_func;
};

/// @brief Function type for ROM assembly. Called at each quadrature point.
///
/// Signature:
///
/// void AssembleLocal(double weight, double hK,
///   const ReducedAssemblingData& data);
///
/// When called:
///   * weight is the quadrature weight associated with the current point
///   * hK is a measure of the local cell's size
///   * data contains various necessary assembly buffers.
typedef std::function<void(double, double, const ReducedAssemblingData&)> AssembleLocalReduced;

/// @brief Generic ROM assembler class.
template<int d>
class ReducedAssembling
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a ROM assembler with possibly distinct test and
    ///        ansatz spaces.
    ReducedAssembling(std::vector<std::shared_ptr<const FESpace>> test_spaces,
      std::vector<std::shared_ptr<const FESpace>> ansatz_spaces,
      std::vector<std::vector<std::shared_ptr<const FESpace>>> aux_spaces,
      int test_dim, int ansatz_dim,
      const double* const* test_basis, const double* const* ansatz_basis,
      const MultiIndex_vector& test_derivatives,
      const MultiIndex_vector& ansatz_derivatives);

    /// @brief Construct a ROM assembler with identical test and ansatz spaces.
    ReducedAssembling(std::vector<std::shared_ptr<const FESpace>> spaces,
      std::vector<std::vector<std::shared_ptr<const FESpace>>> aux_spaces,
      int dim, const double* const* basis,
      const MultiIndex_vector& derivatives);

    /// @brief Add an FE function in the reduced test or ansatz space.
    void AddReducedFunction(const double* func,
      const MultiIndex_vector& derivatives, bool is_test = false);

    /// @brief Add an FE function in the full-order test or ansatz space.
    void AddFullFunction(const double* func,
      const MultiIndex_vector& derivatives, bool is_test = false);

    /// @brief Add an FE function in an auxiliary full-order space.
    void AddAuxFunction(const double* func,
      const MultiIndex_vector& derivatives, int aux_space = 0);

    /// @brief Add an assembly function with scale information.
    void AddAssembly(AssembleLocalReduced assembly, double m = 1.0);

    /// @brief Clear assembly data.
    void Reset();

    /// @brief Perform the actual assembly.
    void Assemble(double multiplier, double** matrix, double* rhs,
      bool clear = false, double* val = nullptr);

  protected:
    /// @brief Are test and ansatz spaces and bases distinct?
    bool DistinctBases();

    /// @brief Helper function to set up the derivative lookup for a given
    ///        set of requested and retrieved multi-indices, to be called only
    ///        from SetDerivativeLookups().
    void SetDerivativeLookup(const MultiIndex_vector& derivatives,
      const MultiIndex_vector& indices, std::vector<int>& lookup);

    /// @brief Set up the derivative lookup structures for the bases and FE functions.
    void SetDerivativeLookups();

    /// @brief Set up the buffers and lookup pointers for the bases and FE functions.
    void SetBasisAndFunctionStorage();

    /// @brief Assemble the reduced test basis at cell c, quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalTestBasis(int c, int p);

    /// @brief Assemble the reduced ansatz basis at cell c, quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalAnsatzBasis(int c, int p);

    /// @brief Assemble all reduced-order FE functions at the current
    ///        quadrature point.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalReducedFunctions();

    /// @brief Assemble all full-order FE functions at cell c,
    ///        quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalFullFunctions(int c, int p);

    /// @brief Assemble all auxiliary FE functions at cell c,
    ///        quadrature point p.
    ///        To be called only from the inner loop of Assemble().
    void AssembleLocalAuxFunctions(int c, int p);

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

        /// @brief For non-aux functions, determines whether this function
        ///        lives in the test or ansatz space.
        const bool is_test;
    };

    /// @brief Test space. May have multiple components.
    const std::vector<std::shared_ptr<const FESpace>> test_spaces;

    /// @brief Ansatz space. May have multiple components.
    const std::vector<std::shared_ptr<const FESpace>> ansatz_spaces;

    /// @brief Auxiliary spaces. May have multiple components each.
    const std::vector<std::vector<std::shared_ptr<const FESpace>>> aux_spaces;

    /// @brief Number of components in the test space.
    const int test_vector_dim;

    /// @brief Number of components in the ansatz space.
    const int ansatz_vector_dim;

    /// @brief Number of basis functions in the reduced test space.
    const int test_dim;

    /// @brief Number of basis functions in the reduced ansatz space.
    const int ansatz_dim;

    /// @brief Test basis nodal vectors.
    const double* const* test_basis;

    /// @brief Ansatz basis nodal vectors.
    const double* const* ansatz_basis;

    /// @brief Which derivatives of the reduced-order test basis do we need?
    const MultiIndex_vector test_derivatives;

    /// @brief Which derivatives of the reduced-order ansatz basis do we need?
    const MultiIndex_vector ansatz_derivatives;

    /// @brief Reduced-order FE functions.
    std::vector<FunctionInfo> reduced_functions;

    /// @brief Full-order FE functions.
    std::vector<FunctionInfo> full_functions;

    /// @brief Auxiliary FE functions for each auxiliary FE space.
    std::vector<std::vector<FunctionInfo>> aux_functions;

    /// @brief Assembly functions to call.
    std::vector<AssembleLocalReduced> assembly_functions;

    /// @brief Multiplier for each assembly function.
    std::vector<double> assembly_multipliers;

    /// @brief FE function values, indexed as [function index][vector dimension, functional]
    std::vector<MultiVector<double, 2>> reduced_values, full_values, aux_values;

    /// @brief Basis function values, indexed as [vector dimension, functional, basis function]
    MultiVector<double, 3> test_values, ansatz_values;

    /// @brief Do we need the test FE space's second derivatives?
    bool test_sd;

    /// @brief Do we need the ansatz FE space's second derivatives?
    bool ansatz_sd;

    /// @brief Do we need each auxiliary FE space's second derivatives?
    std::vector<bool> aux_sd;

    /// @brief Derivatives of the full-order test/ansatz basis to retrieve.
    MultiIndex_vector test_indices, ansatz_indices;

    /// @brief Mapping of the requested derivatives to the retrieved for the
    ///        reduced-order basis functions.
    std::vector<int> test_derivative_lookup, ansatz_derivative_lookup;

    /// @brief Derivatives of each full-order auxiliary basis to retrieve.
    std::vector<MultiIndex_vector> aux_indices;

    /// @brief Mapping of the requested derivatives to the retrieved for the
    ///        auxiliary FE functions.
    std::vector<std::vector<std::vector<int>>> aux_derivative_lookup;

    /// @brief Mapping of the requested derivatives to the retrieved
    ///        for the reduced-order FE functions.
    std::vector<std::vector<int>> reduced_derivative_lookup;

    /// @brief Mapping of the requested derivatives to the retrieved
    ///        for the full-order FE functions.
    std::vector<std::vector<int>> full_derivative_lookup;

    /// @brief Buffer to store all full-order basis values for the current
    ///        cell during assembly.
    std::vector<std::vector<double**>> orig_basis_values;

    /// @brief Buffer to store all spaces' FiniteElement for the current cell
    ///        during assembly.
    std::vector<const FiniteElement*> elements;

    /// @brief Buffer to store all spaces' BaseFunctions for the current cell
    ///        during assembly.
    std::vector<const BaseFunctions*> bf;

    /// @brief Offset of each auxiliary FE space into the basis, FiniteElement
    ///        and BaseFunctions buffers.
    std::vector<unsigned int> aux_basis_offset;

    bool setup_finished;
};

namespace ReducedBasisLookup
{
  /// @brief If the basis is in default order, returns v_k
  const Slice<double> DefaultBasisValues(const MultiVector<double, 3>& basis, int k = 0);

  /// @brief If the basis is in default order, returns d/dx_m v_k
  const Slice<double> DefaultBasisGradients(const MultiVector<double, 3>& basis, int k, int m);

  /// @brief If the basis is in default order, returns d/dx_m v_0
  const Slice<double> DefaultBasisGradients(const MultiVector<double, 3>& basis, int m);

  /// @brief If the function is in default order, returns u_k
  double DefaultFEValue(const MultiVector<double, 2>& func, int k = 0);

  /// @brief If the function is in default order, returns d/dx_m u_k
  double DefaultFEGradient(const MultiVector<double, 2>& func, int k, int m);

  /// @brief If the function is in default order, returns d/dx_m u_0
  double DefaultFEGradient(const MultiVector<double, 2>& func, int m);
}

#endif