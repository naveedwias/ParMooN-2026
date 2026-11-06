// =======================================================================
// @(#)Upwind.C        1.6 11/24/99
//
// Purpose:     upwind stabilization
//
// Authors:     Volker Behns / Gunar Matthies  18.10.99
//              Volker John, bug fixes 18. - 22.10.99
//
// =======================================================================
#ifdef __2D__

#include <Database.h>
#include <FEFunction2D.h>
#include <LinAlg.h>
#include <SquareMatrix2D.h>
#include <Upwind.h>
#include <IsoBoundEdge.h>
#include <BoundComp.h>
#include "BaseCell.h"

#include <stdlib.h>
#include <string.h>
#include <MooNMD_Io.h>

void GetUpwindValue(double UPWIND_ORDER,double t,double *phi)
{
  double s;

  if (UPWIND_ORDER==1)                            // Samarskij upwind
  {
    if (t<0)
      *phi = 1/(2*(1-t));
    else
      *phi = (1+2*t)/(2*(1+t));
  }
  else
  {
    if (UPWIND_ORDER==0)                          // sharp upwind
      *phi = (sgn(t)+1)/2;
    else
    {
      s = std::abs(t);
      *phi = 0.5+0.5*sgn(t)*std::pow(s/(s+1), UPWIND_ORDER);
    }
  }
  return;
}


// =======================================================================
// do upwind assembling for first order nonconforming elements
// =======================================================================
void UpwindForNavierStokes(const CoeffFct2D& Coeff, TSquareMatrix2D *sqmatrix,
const TFEFunction2D *u1, const TFEFunction2D *u2)
{
  double RE;
  double UPWIND_ORDER=TDatabase::ParamDB->UPWIND_ORDER;
  double UPWIND_FLUX_DAMP=TDatabase::ParamDB->UPWIND_FLUX_DAMP;
  int  UPW_APPL = TDatabase::ParamDB->UPWIND_APPLICATION;
  int i,j,k,l,m,n, N_Cells, N_Vertices;
  const double *u1vect, *u2vect;
//  double FctValNQuad[4][4] =
//  {
//    {
//      0.5, 0, 0, 0.5
//    }
//    ,
//    {
//      0.5, 0.5, 0, 0
//    },
//    { 0, 0.5, 0.5, 0 },
//    {
//      0, 0, 0.5, 0.5
//    }
//  };
  double FctValNTria[3][3] =
  {
    {
      0.666666666666666666667,
      -0.333333333333333333333,
      0.666666666666666666667
    },
    {
      0.666666666666666666667,
      0.666666666666666666667,
      -0.333333333333333333333
    },
    {
      -0.333333333333333333333,
      0.666666666666666666667,
      0.666666666666666666667
    }
  };
//  double FctValCQuad[4][4] =
//  {
//    {
//      0.375, 0.375, 0.125, 0.125
//    },
//    { 0.125, 0.375, 0.125, 0.375 },
//    { 0.125, 0.125, 0.375, 0.375 },
//    { 0.375, 0.125, 0.375, 0.125 }
//  };
//  double FctValCTria[3][3] =
//  {
//    {
//      0.416666666666666666667,
//      0.416666666666666666667,
//      0.166666666666666666667
//    },
//    {
//      0.166666666666666666667,
//      0.416666666666666666667,
//      0.416666666666666666667
//    },
//    {
//      0.416666666666666666667,
//      0.166666666666666666667,
//      0.416666666666666666667
//    }
//  };
  double matrix[4][4];
  double val1, val2, val, flux;
  double xcoords[4], ycoords[4];
  double sx, sy, t, phi;
  double nx, ny;

  double *coeffs, *params;
  const int *ColInd, *RowPtr;
  double *Entries;
  int ActiveBound, DirichletBound, end;
  FE_type CurrentElement;

  coeffs = new double[6];
  params = new double[9];
  memset(params, 0, 9 * sizeof(double));

  // get pointers to columns, rows and entries of matrix A
  ColInd = sqmatrix->get_vector_columns();
  RowPtr = sqmatrix->get_row_ptr();
  Entries = sqmatrix->GetEntries();

  // get finite element space for velocity
  auto fespace = u1->GetFESpace2D();

  // get number of free dof
  ActiveBound = fespace->get_n_active();
  // get start of dirichlet nodes in dof array
  DirichletBound = fespace->get_n_active_non_hanging();

  // get vectors of current velocity
  u1vect = u1->GetValues();
  u2vect = u2->GetValues();

  // get collection and number of cells
  auto coll = fespace->GetCollection();
  N_Cells = coll->GetN_Cells();

  // loop over all cells for computing the upwind stabilization
  for(i=0;i<N_Cells;i++)
  {
    // next cell
    auto cell = coll->GetCell(i);
    // pointer to global indices of dof connected with this cell
    auto DOF = fespace->GetGlobalDOF(i);
    // # of vertices
    N_Vertices = cell->GetN_Vertices();
    // get coordinates of the vertices
    for(k=0;k<N_Vertices;k++)
      cell->GetVertex(k)->GetCoords(xcoords[k], ycoords[k]);

    // calculate the coordinates of the center of gravity
    sx = sy = 0.0;
    for(k=0;k<N_Vertices;k++)
    {
      sx += xcoords[k];
      sy += ycoords[k];
    }
    sx /= N_Vertices;
    sy /= N_Vertices;

    // reset local matrix
    memset(matrix, 0, sizeof(double)*16);
    CurrentElement = fespace->get_fe_type(i);

    switch(CurrentElement)
    {
      // ********** nonconforming discretization *******************
      case N_P1_2D_T_A:
      case N_Q1_2D_Q_A:
      case N_Q1_2D_Q_M:
        // for all edges (nodes)
        for(k=0;k<N_Vertices;k++)
        {
          // j : edge which is 'left'
          j = (k+N_Vertices-1) % N_Vertices;
          // normal to edge of dual domain
          nx = sy-ycoords[k];
          ny = xcoords[k]-sx;
          // compute current velocity in center of edge of dual
          // boundary
          switch(N_Vertices)
          {
            // triangle
            case 3: val1 = 0.0;
            val2 = 0.0;
            for(l=0;l<N_Vertices;l++)
            {
              m = DOF[l];
              val = FctValNTria[k][l];
              val1 += u1vect[m]*val;
              val2 += u2vect[m]*val;
            }                                     // endfor l
            break;
            // quadrangle
            case 4: m=DOF[k];
            n=DOF[j];
            val1 = 0.5*(u1vect[m]+u1vect[n]);
            val2 = 0.5*(u2vect[m]+u2vect[n]);
            break;
          }                                       // endswitch
          // compute flux across edge of dual boundary
          flux = val1*nx + val2*ny;
          // get Reynolds number
          Coeff(1, &sx ,&sy, &params, &coeffs);
          RE = 1.0/coeffs[0];
          // parameter for upwind function
          t = UPWIND_FLUX_DAMP*flux*RE*0.5;
          // compute upwind value
          GetUpwindValue(UPWIND_ORDER,t,&phi);
          // compute matrix entries
          switch(UPW_APPL)
          {
            case 0:
              matrix[k][k] += flux*phi;
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(1-phi);
              break;
            case 1:
              matrix[k][k] += flux*(-0.5+phi);
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(0.5-phi);
              break;
            default:
              Output::print("Wrong Upwind Application !!! Default is used !!!");
              matrix[k][k] += flux*phi;
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(1-phi);
              break;
          }
        }                                         // endfor k
        break;

      default:
        ErrThrow("Upwind works only for first order nonconforming elements!");
    }                                             // endswitch

    for(m=0;m<N_Vertices;m++)
    {
      l=DOF[m];
      // cout << "DOF: " << l << endl;
      if(l<ActiveBound)
      {
        // node l is inner or Neumann node
        end=RowPtr[l+1];
        for(n=RowPtr[l];n<end;n++)
        {
          for(k=0;k<N_Vertices;k++)
          {
            if(DOF[k] == ColInd[n])
            {
              // cout << m << "   " << k << endl << n << endl;
              Entries[n] += matrix[m][k];
            }                                     // endif
          }                                       // endfor k
        }                                         // endfor n
      }                                           // endif l
      else
      {
        if(l<DirichletBound)
        {
          // hanging node
          /*

          // implement later
          l -= ActiveBound;
          end = HangingRowPtr[l+1];
          for(n=HangingRowPtr[l];n<end;n++)
          {
            for(k=0;k<N_;k++)
            {
              if(DOF[k] == HangingColInd[n])
              {
          CurrentHangingEntries[n] += MatrixRow[k];
          } // endif
          } // endfor k
          } // endfor n
          */
        }
        else
        {
          // Dirichlet node => do nothing
        }
      }
    }                                             // endfor m
  }                                               // endfor i
  delete params;
  delete coeffs;
}                                                 // end of UpwindForNavierStokes


void UpwindForConvDiff(const CoeffFct2D& Coeff, TSquareMatrix2D *sqmatrix, 
                       double *, const TFESpace2D *fespace,
                       const TFEFunction2D *u1, const TFEFunction2D *u2, 
                       bool ConvIsVelo)
{
  double RE;
  double UPWIND_ORDER=TDatabase::ParamDB->UPWIND_ORDER;
  double UPWIND_FLUX_DAMP=TDatabase::ParamDB->UPWIND_FLUX_DAMP;
  int  UPW_APPL = TDatabase::ParamDB->UPWIND_APPLICATION;
  int i,j,k,l,m,n, N_Cells, N_Vertices;
  int n_mat, m_mat;
  double matrix[4][4];
  double flux;
  double xcoords[4], ycoords[4];
  double sx, sy, t, phi;
  double nx, ny, x, y;
  double nxs[4], nys[4];
  const int *ColInd, *RowPtr;
  double *Entries;
  int ActiveBound, DirichletBound, end;
  double *coeff = new double[6], *param = new double[9];
  FE_type CurrentElement;

  memset(param, 0, 9 * sizeof(double));

  // get pointers to columns, rows and entries of matrix A
  ColInd = sqmatrix->get_vector_columns();
  RowPtr = sqmatrix->get_row_ptr();
  Entries = sqmatrix->GetEntries();
  
  // get number of free dof
  ActiveBound = fespace->get_n_active();
  // get start of dirichlet nodes in dof array
  DirichletBound = fespace->get_n_active_non_hanging();

  // get collection and number of cells
  auto coll = fespace->GetCollection();
  N_Cells = coll->GetN_Cells();

  // loop over all cells for computing the upwind stabilization
  for(i=0;i<N_Cells;i++)
  {
    // next cell
    auto cell = coll->GetCell(i);
    // pointer to global indices of dof connected with this cell
    auto DOF = fespace->GetGlobalDOF(i);
    // # of vertices
    N_Vertices = cell->GetN_Vertices();
    // get coordinates of the vertices
    for(k=0;k<N_Vertices;k++)
      cell->GetVertex(k)->GetCoords(xcoords[k], ycoords[k]);

    CurrentElement = fespace->get_fe_type(i);

    // calculate the coordinates of the center of gravity
    sx = sy = 0.0;
    for(k=0;k<N_Vertices;k++)
    {
      sx += xcoords[k];
      sy += ycoords[k];
    }
    sx /= N_Vertices;
    sy /= N_Vertices;

    // reset local matrix
    memset(matrix, 0, sizeof(double)*16);

    switch(CurrentElement)
    {
      // ********** nonconforming discretization *******************
      case N_P1_2D_T_A:
      case N_Q1_2D_Q_A:
      case N_Q1_2D_Q_M:
        // for all edges (nodes)
        for(k=0;k<N_Vertices;k++)
        {
          // j : edge which is 'left'
          j = (k+N_Vertices-1) % N_Vertices;
          // normal to edge of dual domain
          nx = sy-ycoords[k];
          ny = xcoords[k]-sx;
          // midpoints of edge of dual boundary
          x = (xcoords[k]+sx)*0.5;
          y = (ycoords[k]+sy)*0.5;
          // compute convection in center of edge of dual
          // boundary
          if (!ConvIsVelo)
            Coeff(1, &x, &y, &param, &coeff);
          else
          {
            u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
            u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
          }
          // compute flux across edge of dual boundary
          flux = coeff[1]*nx + coeff[2]*ny;
          // get diffusion coefficient
          Coeff(1, &sx ,&sy, &param, &coeff);
          RE = 1.0/coeff[0];
          // parameter for upwind function
          t = UPWIND_FLUX_DAMP*flux*RE*0.5;
          // compute upwind value
          GetUpwindValue(UPWIND_ORDER,t,&phi);
          // compute matrix entries
          switch(UPW_APPL)
          {
            case 0:
              matrix[k][k] += flux*phi;
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(1-phi);
              break;
            case 1:
              matrix[k][k] += flux*(-0.5+phi);
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(0.5-phi);
              break;
            default:
              Output::print("Wrong Upwind Application !!! Default is used !!!");
              matrix[k][k] += flux*phi;
              matrix[k][j] -= flux*phi;
              matrix[j][k] += flux*(1-phi);
              matrix[j][j] -= flux*(1-phi);
              break;
          }
        }                                         // endfor k
        break;
        // ********** conforming discretization *******************
      case C_P1_2D_T_A:
      case C_Q1_2D_Q_A:
      case C_Q1_2D_Q_M:
        if (N_Vertices==3)
        {
          for (k=0;k<3;k++)
          {
            // centre of face (P[k]P[(k+1)%3])
            x=(xcoords[k]+xcoords[(k+1)%3])*0.5;
            y=(ycoords[k]+ycoords[(k+1)%3])*0.5;
            // normal of dual face directed to
            nxs[(2*k+1)%3] = sy-y;
            // P[(k+1)%3]
            nys[(2*k+1)%3] = x-sx;
          }
          // assemble matrix, compute integral
          for(m=0;m<N_Vertices;m++)
          {
            for(n=0;n<N_Vertices;n++)
            {
              if (n==m)
              {                                   // upwind to point P[(m-1)%3]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[(m+2)%3])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[(m+2)%3])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of
                  Coeff(1, &x, &y, &param, &coeff);
                // the underlying pde
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = -(coeff[1]*nxs[(2*m+2)%3]+coeff[2]*nys[(2*m+2)%3]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                if (UPW_APPL==1)
                  phi += 0.5;
                matrix[m][n] += -flux*(1.0-phi);
                // upwind to point P[(m+1)%3]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[(m+1)%3])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[(m+1)%3])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = (coeff[1]*nxs[(2*m+1)%3]+coeff[2]*nys[(2*m+1)%3]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                if (UPW_APPL==1)
                  phi += 0.5;
                matrix[m][n] += -flux*(1.0-phi);
              }
              else
              {                                   // upwind to point P[n]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[n])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[n])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = (coeff[1]*nxs[(m+n)%3]+coeff[2]*nys[(m+n)%3]);
                if (m==(n+1)%3) flux = -flux;
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                matrix[m][n] += flux*(1.0-phi);
              }
            }                                     // end for n
          }                                       // end for m
        }
        else
        {
          for (k=0;k<4;k++)
          {
            // centre of face (P[k]P[(k+1)%4])
            x=(xcoords[k]+xcoords[(k+1)%4])*0.5;
            y=(ycoords[k]+ycoords[(k+1)%4])*0.5;
            nxs[k] = sy-y;                        // normal of dual face directed to
            nys[k] = x-sx;                        // P[(k+1)%4]
          }
          // assemble matrix, compute integral
          for(m=0;m<N_Vertices;m++)
          {
            if (m<2) m_mat=m;
            if (m==2) m_mat=3;
            if (m==3) m_mat=2;
            for(n=0;n<N_Vertices;n++)
            {
              if (n<2) n_mat=n;
              if (n==2) n_mat=3;
              if (n==3) n_mat=2;
              if (n==m)
              {                                   // upwind to point P[(m+3)%4]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[(m+3)%4])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[(m+3)%4])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                //Output::print(coeff[1], " ", coeff[2]);
                flux = -(coeff[1]*nxs[(m+3)%4]+coeff[2]*nys[(m+3)%4]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                matrix[m_mat][n_mat] += -flux*(1.0-phi);
                // upwind to point P[(m+1)%4]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[(m+1)%4])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[(m+1)%4])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = (coeff[1]*nxs[m%4]+coeff[2]*nys[m%4]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                matrix[m_mat][n_mat] += -flux*(1.0-phi);
              }
              if(n==(m+1)%4)
              {                                   // upwind to point P[(m+1)%4]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[n])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[n])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = (coeff[1]*nxs[m%4]+coeff[2]*nys[m%4]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                matrix[m_mat][n_mat] += flux*(1.0-phi);
              }
              if(n==(m+3)%4)
              {                                   // upwind to point P[(m+3)%4]
                // coordinates on dual boundary
                x= (xcoords[m]+xcoords[n])*0.25+sx*0.5;
                // coordinates on dual boundary
                y= (ycoords[m]+ycoords[n])*0.25+sy*0.5;
                if (!ConvIsVelo)
                  // get parameter function of the underlying pde
                  Coeff(1, &x, &y, &param, &coeff);
                else
                {
                  u1->FindGradientLocal(cell,i,x,y,&coeff[1]);
                  u2->FindGradientLocal(cell,i,x,y,&coeff[2]);
                }
                flux = -(coeff[1]*nxs[n%4]+coeff[2]*nys[n%4]);
                // get diffusion coefficient
                Coeff(1, &sx ,&sy, &param, &coeff);
                RE = 1.0/coeff[0];
                // parameter for upwind function
                t= UPWIND_FLUX_DAMP*flux*RE*0.5;
                GetUpwindValue(UPWIND_ORDER,t,&phi);
                matrix[m_mat][n_mat] += flux*(1.0-phi);
              }
            }                                     // end for n
          }                                       // end for m
        }
        break;

      default:
        ErrThrow("Upwind works only for first order elements!");
        break;
    }                                             // endswitch
    for(m=0;m<N_Vertices;m++)
    {
      l = DOF[m];
      // cout << "DOF: " << l << endl;
      if (l < ActiveBound)
      {
        // node l is inner or Neumann node
        end = RowPtr[l+1];
        for(n=RowPtr[l];n<end;n++)
        {
          for(k=0;k<N_Vertices;k++)
          {
            if(DOF[k] == ColInd[n])
            {
              // cout << m << "   " << k << endl << n << endl;
              Entries[n] += matrix[m][k];
            }                                     // endif
          }                                       // endfor k
        }                                         // endfor n
      }                                           // endif l
      else
      {
        if (l < DirichletBound)
        {
          // hanging node
          /*

          // implement later
          l -= ActiveBound;
          end = HangingRowPtr[l+1];
          for(n=HangingRowPtr[l];n<end;n++)
          {
            for(k=0;k<N_;k++)
            {
              if(DOF[k] == HangingColInd[n])
              {
          CurrentHangingEntries[n] += MatrixRow[k];
          } // endif
          } // endfor k
          } // endfor n
          */
        }
        else
        {
          // Dirichlet node => do nothing
        }
      }
    }                                             // endfor m
  }                                               // endfor i
  delete [] param;
  delete [] coeff;
}                                                 // UpwindForConvDiff
#endif // #ifdef __2D__
