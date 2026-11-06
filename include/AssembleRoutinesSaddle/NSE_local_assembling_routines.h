#ifndef INCLUDE_ASSEMBLEROUTINESSADDLE_NSE_LOCAL_ASSEMBLING_ROUTINES_H
#define INCLUDE_ASSEMBLEROUTINESSADDLE_NSE_LOCAL_ASSEMBLING_ROUTINES_H

///////////////////////////////////////////////////////////////////////////////
// standard terms, linear (needed for Galerkin)

template<int d>
void NSResistanceMassMatrixSingle(double Mult, const double *coeff,
                                  const double *param, double hK,
                                  const double **OrigValues,
                                  const int *N_BaseFuncts,
                                  double ***LocMatrices, double **LocRhs);

template<int d>
void NSResistanceMassMatrix(double Mult, const double *coeff,
                            const double *param, double hK,
                            const double **OrigValues, const int *N_BaseFuncts,
                            double ***LocMatrices, double **LocRhs);

template <int d>
void NSLaplaceGradGradSingle(double Mult, const double *coeff,
                             const double *param, double hK,
                             const double**OrigValues, const int *N_BaseFuncts,
                             double ***LocMatrices, double **LocRhs);
template <int d>
void NSLaplaceGradGrad(double Mult, const double *coeff, const double *param,
                       double hK, const double**OrigValues,
                       const int *N_BaseFuncts,double ***LocMatrices,
                       double **LocRhs);
template <int d>
void NSLaplaceDeformation(double Mult, const double *coeff, const double *param,
                          double hK, const double**OrigValues,
                          const int *N_BaseFuncts, double ***LocMatrices,
                          double **LocRhs);
template <int d>
void NSDivergenceBlocks(double Mult, const double *coeff, const double *param,
                        double hK, const double**OrigValues,
                        const int *N_BaseFuncts, double ***LocMatrices,
                        double **LocRhs, int sign=1);
template <int d>
void NSGradientBlocks(double Mult, const double *coeff, const double *param,
                      double hK, const double**OrigValues,
                      const int *N_BaseFuncts, double ***LocMatrices,
                      double **LocRhs);
template <int d>
void NSRightHandSide(double Mult, const double *coeff, const double *param,
                     double hK, const double **OrigValues,
                     const int *N_BaseFuncts, double ***LocMatrices,
                     double **LocRhs, int sign = 1);

///////////////////////////////////////////////////////////////////////////////
// local assembling for H(div)-conforming elements (discontinuous Galerkin)
template <int d>
void NSLaplaceGradGrad_dg(double Mult, const double *coeff, const double *param,
                          double hK, const double**OrigValues,
                          const int *N_BaseFuncts,double ***LocMatrices,
                          double **LocRhs);

///////////////////////////////////////////////////////////////////////////////
// local assemblings for the nonlinear term


// convective:
//
// n(u; v, w) = ((u \cdot \nabla v), w)

template <int d>
void NSNonlinearTerm_convective_Single(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

template <int d>
void NSNonlinearTerm_convective(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

// skew symmetric form:
//
// n(u; v, w) = ( ((u \cdot \nabla v), w) - ((u \cdot \nabla w), v) )/2

template <int d>
void NSNonlinearTerm_convective_dg(
	double Mult, const double *, const double *param, double, const double **OrigValues,
       const int *N_BaseFuncts, double ***LocMatrices, double **);  
template <int d>
void NSNonlinearTerm_skew_symmetric_Single(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

template <int d>
void NSNonlinearTerm_skew_symmetric(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

// rotational form:
//
// n(u; v, w) = (((\nabla \times v) \times u), w)
//
// notably *not*
//
// n(u; v, w) = (((\nabla \times u) \times v), w)

template <int d>
void NSNonlinearTerm_rotational(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

// Energy, Momentum, and Angular momentum Conserving (EMAC):
//
// n(u; v, w) = 2 ((\nabla v)^T u, w) + ((\nabla \cdot u) v, w)
//
// notably *not*
//
// n(u; v, w) = 2 ((\nabla u) v, w) + ((\nabla \cdot u) v, w)

template <int d>
void NSNonlinearTerm_emac(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

// divergence form:
//
// n(u; v, w) = ((u \cdot \nabla v), w) + ((\nabla \cdot u) v, w) / 2

template <int d>
void NSNonlinearTerm_divergence(
  double Mult, const double *, const double *param, double,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **);

///////////////////////////////////////////////////////////////////////////////
// more special terms

template <int d>
void NSCoriolis(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

template <int d>
void NSRightHandSideExplicitNL(
  double Mult, const double *coeff, const double *param, double hK,
  const double **OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **LocRhs);

///////////////////////////////////////////////////////////////////////////////
// stabilization terms

// Grad-Div
template <int d>
void NSGradDiv(double Mult, const double *coeff, const double *param, double hK,
               const double **OrigValues, const int *N_BaseFuncts,
               double ***LocMatrices, double **LocRhs, double delta0,
               double characteristic_length);

template <int d>
void NSGradDiv_RightHandSide(double Mult, const double *coeff,
                             const double *param, double hK,
                             const double **OrigValues,
                             const int *N_BaseFuncts, double ***LocMatrices,
                             double **LocRhs, double delta0,
                             double characteristic_length);

// PSPG (Pressure stabilization Petrov-Galerkin)
double compute_PSPG_delta(double delta0, double hK, double nu);
template <int d>
void NSPSPG(double Mult, const double *coeff, const double *param, double hK,
            const double **OrigValues, const int *N_BaseFuncts,
            double ***LocMatrices, double **LocRhs, double delta0);
template <int d>
void NSPSPG_RightHandSide(double Mult, const double *coeff, const double *param,
                          double hK, const double **OrigValues,
                          const int *N_BaseFuncts, double ***LocMatrices,
                          double **LocRhs, double delta0);

// symmetric GLS (Galerkin least-squares) method
double compute_GLS_delta(double delta0, double hK, double nu,
                         double sigma = 0.0, double characteristic_length = 0.1);

template <int d>
void NSsymmGLS(double Mult, const double *coeff, const double *param, double hK,
               const double **OrigValues, const int *N_BaseFuncts,
               double ***LocMatrices, double **LocRhs, double delta0);
template <int d>
void NSsymmGLS_RightHandSide(double Mult, const double *coeff, const double *param,
                             double hK, const double **OrigValues,
                             const int *N_BaseFuncts, double ***LocMatrices,
                             double **LocRhs, double delta0);

// non-symmetric GLS (Galerkin least-squares) method
template <int d>
void NSnonsymmGLS(double Mult, const double *coeff, const double *param,
                  double hK, const double **OrigValues, const int *N_BaseFuncts,
                  double ***LocMatrices, double **LocRhs, double delta0,
                  double characteristic_length = 0.1);
template <int d>
void NSnonsymmGLS_RightHandSide(double Mult, const double *coeff,
                                const double *param, double hK,
                                const double **OrigValues,
                                const int *N_BaseFuncts, double ***LocMatrices,
                                double **LocRhs, double delta0,
                                double characteristic_length = 0.1);

template <int d>
void NS_BrezziPitkaeranta(double Mult, const double *coeff, const double *param,
                          double hK, const double **OrigValues,
                          const int *N_BaseFuncts, double ***LocMatrices,
                          double **LocRhs, double delta0);

///////////////////////////////////////////////////////////////////////////////
template <int d>
void OseenSingle(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **);

template <int d>
void OseenSingleSUPG(
  double Mult, const double *coeff, const double *, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **, double delta0, double delta1);

template <int d>
void OseenGradBlockSUPG(double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **,double delta0);

template <int d>
void OseenRhsSUPG(double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***,
  double **LocRhs, double delta0);
///////////////////////////////////////////////////////////////////////////////
// routines to pass parameters (values of fe functions) to local assembling
// routines
template <int d>
void NSLaplaceDeformationSUPG(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **, double delta0, double delta1);

template <int d>
void NSGradientBlocksSUPG(double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **,double delta0);

template <int d>
void NSRightHandSideSUPG(double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***,
  double **LocRhs, double delta0);

///////////////////////////////////////////////////////////////////////////////
// routines to pass parameters (values of fe functions) to local assembling
// routines

// velocity
template <int d>
void NSParamsVelocity(const double *in, double *out);

// velocity *and* gradient
template <int d>
void NSParamsVelocityDerivatives(const double *in, double *out);

// velocity gradient only
template <int d>
void NSParamsVelocityGradient(const double *in, double *out);


#endif // INCLUDE_ASSEMBLEROUTINESSADDLE_NSE_LOCAL_ASSEMBLING_ROUTINES_H

///////////////////////////////////////////////////////////////////////////////
// SUPG/PSPG/grad-div stabilized method for the Oseen and the stationary
// Navier-Stokes problem (space_discretization_type "supg_pspg_graddiv"),
// see M. Afzal, N. Ahmed, V. John, Comput. Math. Appl. 223 (2026) 467-486,
// Eqs. (7)-(8) and (11)-(12). Only NSTYPE 14 (A, B, B^T, C blocks) and 2D.
//
//   A((u,p),(v,q)) = nu(grad u, grad v) + ((b.grad)u + sigma u, v) - (div v, p)
//                  + (div u, q) + sum_K mu_K (div u, div v)_K
//                  + sum_K (-nu Lap u + (b.grad)u + sigma u + grad p,
//                           delta_K ((b.grad)v + grad q))_K
//   L((v,q))       = (f, v) + sum_K (f, delta_K ((b.grad)v + grad q))_K
//
// ParMooN stores the continuity row with the opposite sign, i.e. the B and C
// blocks and the pressure right-hand side are assembled as -( ... ).
//
// b = coeff[4..5], sigma = coeff[6] for Oseen (oseen == true), b = param[0..1]
// (current velocity iterate) and sigma = 0 for Navier-Stokes.
///////////////////////////////////////////////////////////////////////////////
/// @brief stabilization parameters delta_K and mu_K, (11) and (12) in the paper
/// with delta_0 = spg_delta0 and mu_0 = spg_mu0:
///   inf-sup stable pairs : delta_K = delta_0 h_K^2, mu_K = mu_0
///   equal order pairs    : delta_K = delta_0 h_K (nu < h_K), delta_0 h_K^2 else
///                          mu_K    = mu_0 h_K
void compute_SPG_parameters(double hK, double nu, double delta0, double mu0,
                            bool equal_order, double& deltaK, double& muK);

/// @brief velocity-velocity blocks A11, A12, A21, A22 (Galerkin, SUPG, grad-div)
template <int d>
void NS_SPG_VelocityBlocks(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **, double delta0, double mu0, bool equal_order, bool oseen);

/// @brief pressure-velocity blocks B^T (Galerkin -(div v,p) + SUPG with grad p)
template <int d>
void NS_SPG_GradientBlocks(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **, double delta0, double mu0, bool equal_order, bool oseen);

/// @brief PSPG contributions to B and the C block (the Galerkin part of B is
/// assembled by NSDivergenceBlocks)
template <int d>
void NS_SPG_PressureBlocks(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***LocMatrices,
  double **, double delta0, double mu0, bool equal_order, bool oseen);

/// @brief right-hand side (f, v + delta_K (b.grad)v) and -(g, q) - delta_K (f, grad q)
template <int d>
void NS_SPG_RightHandSide(
  double Mult, const double *coeff, const double *param, double hK,
  const double**OrigValues, const int *N_BaseFuncts, double ***,
  double **LocRhs, double delta0, double mu0, bool equal_order, bool oseen);
