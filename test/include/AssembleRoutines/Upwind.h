// =======================================================================
// @(#)Upwind.h        1.4 10/18/99
//
// Purpose:     upwind stabilization
//              do upwind assembling for first order nonconforming elements
//
// Authors:     Volker Behns / Gunar Matthies  18.10.99
//
// =======================================================================


#ifndef __UPWIND__
#define __UPWIND__

void UpwindForNavierStokes(const CoeffFct2D& Coeff, TSquareMatrix2D *sqmatrix, 
			   const TFEFunction2D *u1, const TFEFunction2D *u2);

void UpwindForConvDiff(const CoeffFct2D& Coeff, TSquareMatrix2D* sqmatrix, 
                       double* RHS, const TFESpace2D* fespace,
                       const TFEFunction2D* u1, const TFEFunction2D* u2,
                       bool ConvIsVelo );
#endif
