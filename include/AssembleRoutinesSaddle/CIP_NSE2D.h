// =============================================================================
// Continuous interior penalty stabilization of the convective derivative for
// the incompressible Navier-Stokes equations with exactly divergence-free
// (Scott-Vogelius) elements, IMEX version with an explicit transport field:
//
//   S(u,v) = N^{-1} sum_j delta_j S_j(u,v),  N = max{ ||b||_{0,inf}, u_min }
//           (optionally N = min{ N, U_max }), S = 0 if N = 0,
//   S_1 = sum_F int_F h_F^2 [ (b.grad)u x n ] [ (b.grad)v x n ],
//   S_2 = sum_F int_F h_F^4 [ B u ] [ B v ],           B w = curl( (b.grad)w ),
//   S_3 = sum_F int_F h_F^6 [ grad B u ] : [ grad B v ],
//
// with b the (continuous, piecewise polynomial) transport field b = 2u^n -
// u^{n-1}, the jumps [.] across the interior edges F, and the 2D curl
// curl w = d_1 w_2 - d_2 w_1, w x n = w_1 n_2 - w_2 n_1.
// (Pressure-robust error estimates for a stabilized BDF2-IMEX Navier-Stokes
// scheme, Sections 2.2 and 3.)
//
// The terms are added to the velocity blocks A11, A12, A21, A22 of the block
// matrix. Affine triangles and Lagrange velocity elements are assumed; the
// basis functions are evaluated on the reference element and transformed with
// the affine map, so that second derivatives (needed for S_2) and third
// derivatives of P2 functions (zero, needed for S_3) are available. S_3 uses
// the exact third derivatives only for P2; for higher order velocity spaces
// the third derivatives of the basis functions are not available and S_3 is
// computed without them (a warning is printed once).
// =============================================================================
#ifndef CIP_NSE2D_H
#define CIP_NSE2D_H

#include "BlockFEMatrix.h"
#include "FEVectFunct2D.h"
#include <array>

struct CIP_parameters
{
  double delta1 = 1e-2;
  double delta2 = 1e-4;
  double delta3 = 1e-6;
  double u_min = 0.25;   // lower bound of the normalization
  double u_max = -1.;    // upper bound (clamp), inactive if <= 0
  int quad_degree = 8;   // degree of the edge quadrature
};

/// @brief add the CIP stabilization S(.,.) with transport field b to the
/// velocity blocks of A. Returns the normalization N used.
double assemble_CIP_NSE2D(BlockFEMatrix& A, const TFEVectFunct2D& b,
                          const CIP_parameters& param);

/// @brief maximum norm of a Lagrange finite element vector function
/// (maximum over the degrees of freedom)
double max_norm(const TFEVectFunct2D& b);

#endif
