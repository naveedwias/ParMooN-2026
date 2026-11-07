#include "CIP_NSE2D.h"
#include "BaseCell.h"
#include "BaseFunctions.h"
#include "Collection.h"
#include "FEDatabase.h"
#include "FESpace2D.h"
#include "FiniteElement.h"
#include "Joint.h"
#include "MooNMD_Io.h"
#include "QuadratureFormulaDatabase.h"
#include "QuadFormula.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

double max_norm(const TFEVectFunct2D& b)
{
  int n = b.GetLength();
  const double* v = b.GetValues();
  double m = 0.;
  for(int i = 0; i < n; ++i)
    m = std::max(m, std::sqrt(v[i]*v[i] + v[i+n]*v[i+n]));
  return m;
}

namespace
{
  // values and derivatives (up to second order) of all basis functions of
  // the element on 'cell' at the physical point (x,y), in physical
  // coordinates; affine triangles only
  struct CellEval
  {
    int n_dof;
    std::vector<double> v, vx, vy, vxx, vxy, vyy;
  };

  void evaluate_basis(const TBaseCell* cell, const FiniteElement& fe,
                      double x, double y, CellEval& out)
  {
    double x0 = cell->GetVertex(0)->GetX(), y0 = cell->GetVertex(0)->GetY();
    double x1 = cell->GetVertex(1)->GetX(), y1 = cell->GetVertex(1)->GetY();
    double x2 = cell->GetVertex(2)->GetX(), y2 = cell->GetVertex(2)->GetY();
    // J = [[a, b],[c, d]], x = x0 + a xi + b eta, y = y0 + c xi + d eta
    double a = x1-x0, bb = x2-x0, c = y1-y0, dd = y2-y0;
    double det = a*dd - bb*c;
    double xi = ( dd*(x-x0) - bb*(y-y0))/det;
    double eta = (-c*(x-x0) + a*(y-y0))/det;
    // inverse transpose: xi_x = dd/det, xi_y = -bb/det, eta_x = -c/det, eta_y = a/det
    double xix = dd/det, xiy = -bb/det, etx = -c/det, ety = a/det;
    const BaseFunctions* bf = fe.GetBaseFunct();
    int n = fe.GetN_DOF();
    out.n_dof = n;
    std::vector<double> r0(n), r10(n), r01(n), r20(n), r11(n), r02(n);
    bf->GetDerivatives(MultiIndex2D::D00, xi, eta, r0.data());
    bf->GetDerivatives(MultiIndex2D::D10, xi, eta, r10.data());
    bf->GetDerivatives(MultiIndex2D::D01, xi, eta, r01.data());
    bf->GetDerivatives(MultiIndex2D::D20, xi, eta, r20.data());
    bf->GetDerivatives(MultiIndex2D::D11, xi, eta, r11.data());
    bf->GetDerivatives(MultiIndex2D::D02, xi, eta, r02.data());
    out.v = r0;
    out.vx.resize(n); out.vy.resize(n);
    out.vxx.resize(n); out.vxy.resize(n); out.vyy.resize(n);
    for(int l = 0; l < n; ++l)
    {
      out.vx[l] = r10[l]*xix + r01[l]*etx;
      out.vy[l] = r10[l]*xiy + r01[l]*ety;
      out.vxx[l] = r20[l]*xix*xix + 2*r11[l]*xix*etx + r02[l]*etx*etx;
      out.vyy[l] = r20[l]*xiy*xiy + 2*r11[l]*xiy*ety + r02[l]*ety*ety;
      out.vxy[l] = r20[l]*xix*xiy + r11[l]*(xix*ety + xiy*etx) + r02[l]*etx*ety;
    }
  }

  // transport field b and its first and second derivatives on one side
  struct FieldEval
  {
    double b[2], bx[2], by[2], bxx[2], bxy[2], byy[2];
  };

  void evaluate_field(const CellEval& ev, const int* DOF, const double* vals,
                      int length, FieldEval& f)
  {
    for(int c = 0; c < 2; ++c)
    {
      f.b[c] = f.bx[c] = f.by[c] = f.bxx[c] = f.bxy[c] = f.byy[c] = 0.;
      for(int l = 0; l < ev.n_dof; ++l)
      {
        double u = vals[DOF[l] + c*length];
        f.b[c] += u*ev.v[l];
        f.bx[c] += u*ev.vx[l];
        f.by[c] += u*ev.vy[l];
        f.bxx[c] += u*ev.vxx[l];
        f.bxy[c] += u*ev.vxy[l];
        f.byy[c] += u*ev.vyy[l];
      }
    }
  }

  // For the basis function psi (index l) times e_c (c = 0,1) compute
  //   t  = (b.grad) psi                       -> T = t e_c
  //   S1: T x n = T_1 n_2 - T_2 n_1  -> c=0: t n_2, c=1: -t n_1
  //   S2: B = curl T = d_1 T_2 - d_2 T_1 -> c=0: -d_2 t, c=1: d_1 t
  //       d_j t = (d_j b).grad psi + b.grad(d_j psi)
  //   S3: grad B (two components) -> c=0: -grad(d_2 t), c=1: grad(d_1 t)
  //       d_i d_j t = (d_i d_j b).grad psi + (d_j b).grad(d_i psi)
  //                 + (d_i b).grad(d_j psi) + b.grad(d_i d_j psi)
  //       third derivatives of psi are zero for P2 and ignored otherwise
  struct Terms
  {
    double s1, s2, s3[2];
  };

  Terms terms(const CellEval& ev, const FieldEval& f, int l, int c, double nx,
              double ny)
  {
    double t = f.b[0]*ev.vx[l] + f.b[1]*ev.vy[l];
    double d1t = f.bx[0]*ev.vx[l] + f.bx[1]*ev.vy[l]
                 + f.b[0]*ev.vxx[l] + f.b[1]*ev.vxy[l];
    double d2t = f.by[0]*ev.vx[l] + f.by[1]*ev.vy[l]
                 + f.b[0]*ev.vxy[l] + f.b[1]*ev.vyy[l];
    // second derivatives of t (third derivatives of psi omitted)
    double d11t = f.bxx[0]*ev.vx[l] + f.bxx[1]*ev.vy[l]
                  + 2*(f.bx[0]*ev.vxx[l] + f.bx[1]*ev.vxy[l]);
    double d12t = f.bxy[0]*ev.vx[l] + f.bxy[1]*ev.vy[l]
                  + f.by[0]*ev.vxx[l] + f.by[1]*ev.vxy[l]
                  + f.bx[0]*ev.vxy[l] + f.bx[1]*ev.vyy[l];
    double d22t = f.byy[0]*ev.vx[l] + f.byy[1]*ev.vy[l]
                  + 2*(f.by[0]*ev.vxy[l] + f.by[1]*ev.vyy[l]);
    Terms r;
    if(c == 0)
    {
      r.s1 = t*ny;
      r.s2 = -d2t;
      r.s3[0] = -d12t;
      r.s3[1] = -d22t;
    }
    else
    {
      r.s1 = -t*nx;
      r.s2 = d1t;
      r.s3[0] = d11t;
      r.s3[1] = d12t;
    }
    return r;
  }
}

double assemble_CIP_NSE2D(BlockFEMatrix& A, const TFEVectFunct2D& b,
                          const CIP_parameters& param)
{
  auto space = b.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  const double* b_values = b.GetValues();
  int length = b.GetLength();

  double norm = std::max(max_norm(b), param.u_min);
  if(param.u_max > 0.)
    norm = std::min(norm, param.u_max);
  if(norm <= 0.)
    return 0.;
  double scale = 1./norm;

  // the velocity blocks A11, A12, A21, A22 (row-major block layout)
  auto blocks = A.get_blocks_uniquely();
  size_t n_blocks_per_row = A.get_n_cell_columns();
  FEMatrix* M[2][2];
  for(int r = 0; r < 2; ++r)
    for(int c = 0; c < 2; ++c)
      M[r][c] = blocks[r*n_blocks_per_row + c].get();
  // (the blocks are returned in row-major order of the block grid; for the
  // Navier-Stokes types the velocity blocks are the first two of each of the
  // first two rows)

  static bool warned = false;
  if(!warned && space->get_fe(0).GetBaseFunct()->GetPolynomialDegree() > 2
     && param.delta3 > 0.)
  {
    Output::warn("CIP_NSE2D", "S_3 is computed without the third derivatives "
                 "of the basis functions (exact only for P2)");
    warned = true;
  }

  const TQuadFormula* qf = QuadratureFormulaDatabase::qf_from_degree(
      param.quad_degree, BFRefElements::BFUnitLine);
  int n_q;
  const double *w_q, *z_q;
  qf->GetFormulaData(n_q, w_q, z_q);

  // index of each cell in the collection
  int n_cells = coll->GetN_Cells();
  std::map<const TBaseCell*, int> cell_index;
  for(int i = 0; i < n_cells; ++i)
    cell_index[coll->GetCell(i)] = i;

  std::map<int, std::map<int, double>> entries[2][2];
  CellEval evp, evm;
  FieldEval fp, fm;
  const int* edge_vertex;

  for(int i = 0; i < n_cells; ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    cell->GetRefDesc()->GetShapeDesc()->GetEdgeVertex(edge_vertex);
    for(int e = 0; e < cell->GetN_Edges(); ++e)
    {
      const TJoint* joint = cell->GetJoint(e);
      const TBaseCell* neigh = joint->GetNeighbour(cell);
      if(neigh == nullptr)
        continue; // boundary edge
      // periodic joints: the neighbour lies at the opposite side of the
      // domain, its points are shifted by the translation between the two
      // copies of the edge
      double shift_x = 0., shift_y = 0.;
      if(joint->GetType() == PeriodicJoint)
      {
        const int* ev_n;
        neigh->GetRefDesc()->GetShapeDesc()->GetEdgeVertex(ev_n);
        for(int en = 0; en < neigh->GetN_Edges(); ++en)
        {
          if(neigh->GetJoint(en) == joint)
          {
            const TVertex* w0 = neigh->GetVertex(ev_n[2*en]);
            const TVertex* w1 = neigh->GetVertex(ev_n[2*en+1]);
            const TVertex* v0 = cell->GetVertex(edge_vertex[2*e]);
            const TVertex* v1 = cell->GetVertex(edge_vertex[2*e+1]);
            shift_x = 0.5*(w0->GetX() + w1->GetX()) - 0.5*(v0->GetX() + v1->GetX());
            shift_y = 0.5*(w0->GetY() + w1->GetY()) - 0.5*(v0->GetY() + v1->GetY());
            break;
          }
        }
      }
      auto it = cell_index.find(neigh);
      if(it == cell_index.end() || it->second < i)
        continue; // each interior edge once (from the cell with smaller index)
      int j = it->second;
      const TVertex* v0 = cell->GetVertex(edge_vertex[2*e]);
      const TVertex* v1 = cell->GetVertex(edge_vertex[2*e+1]);
      double x0 = v0->GetX(), y0 = v0->GetY(), x1 = v1->GetX(), y1 = v1->GetY();
      double nx = y1 - y0, ny = x0 - x1;
      double hF = std::sqrt(nx*nx + ny*ny);
      nx /= hF; ny /= hF;
      double w1 = param.delta1*hF*hF, w2 = param.delta2*std::pow(hF, 4),
             w3 = param.delta3*std::pow(hF, 6);

      const FiniteElement& fep = space->get_fe(i);
      const FiniteElement& fem = space->get_fe(j);
      const int* DOFp = space->GetGlobalDOF(i);
      const int* DOFm = space->GetGlobalDOF(j);
      // union of the global dofs of both cells
      std::map<int, std::pair<int,int>> dofs; // global -> (local +, local -)
      for(int l = 0; l < fep.GetN_DOF(); ++l)
        dofs[DOFp[l]] = {l, -1};
      for(int l = 0; l < fem.GetN_DOF(); ++l)
      {
        auto it2 = dofs.find(DOFm[l]);
        if(it2 != dofs.end())
          it2->second.second = l;
        else
          dofs[DOFm[l]] = {-1, l};
      }
      std::vector<int> gdof;
      for(auto& d : dofs) gdof.push_back(d.first);
      size_t nd = gdof.size();
      // jumps per quad point: [dof][component][term]
      std::vector<Terms> jump(nd*2);
      std::vector<double> loc[2][2];
      for(int r = 0; r < 2; ++r)
        for(int c = 0; c < 2; ++c)
          loc[r][c].assign(nd*nd, 0.);

      for(int q = 0; q < n_q; ++q)
      {
        double s = 0.5*(z_q[q] + 1.);
        double x = x0 + s*(x1-x0), y = y0 + s*(y1-y0);
        double w = 0.5*hF*w_q[q];
        evaluate_basis(cell, fep, x, y, evp);
        evaluate_basis(neigh, fem, x + shift_x, y + shift_y, evm);
        evaluate_field(evp, DOFp, b_values, length, fp);
        evaluate_field(evm, DOFm, b_values, length, fm);
        for(size_t a = 0; a < nd; ++a)
        {
          auto lp = dofs[gdof[a]];
          for(int c = 0; c < 2; ++c)
          {
            Terms jp{0., 0., {0., 0.}}, jm{0., 0., {0., 0.}};
            if(lp.first >= 0) jp = terms(evp, fp, lp.first, c, nx, ny);
            if(lp.second >= 0) jm = terms(evm, fm, lp.second, c, nx, ny);
            Terms& J = jump[a*2 + c];
            J.s1 = jp.s1 - jm.s1;
            J.s2 = jp.s2 - jm.s2;
            J.s3[0] = jp.s3[0] - jm.s3[0];
            J.s3[1] = jp.s3[1] - jm.s3[1];
          }
        }
        for(size_t a = 0; a < nd; ++a)
          for(int r = 0; r < 2; ++r)
          {
            const Terms& Ja = jump[a*2 + r];
            for(size_t bb = 0; bb < nd; ++bb)
              for(int c = 0; c < 2; ++c)
              {
                const Terms& Jb = jump[bb*2 + c];
                loc[r][c][a*nd + bb] += w*scale*(w1*Ja.s1*Jb.s1 + w2*Ja.s2*Jb.s2
                                        + w3*(Ja.s3[0]*Jb.s3[0] + Ja.s3[1]*Jb.s3[1]));
              }
          }
      }
      for(int r = 0; r < 2; ++r)
        for(int c = 0; c < 2; ++c)
          for(size_t a = 0; a < nd; ++a)
            for(size_t bb = 0; bb < nd; ++bb)
            {
              double v = loc[r][c][a*nd + bb];
              if(v != 0.)
                entries[r][c][gdof[a]][gdof[bb]] += v;
            }
    }
  }
  // Dirichlet rows must not be touched (they hold the identity)
  int n_active = space->get_n_active();
  for(int r = 0; r < 2; ++r)
    for(int c = 0; c < 2; ++c)
    {
      for(auto& row : entries[r][c])
      {
        if(row.first >= n_active)
          continue;
        for(auto& col : row.second)
          M[r][c]->add(row.first, col.first, col.second);
      }
    }
  return norm;
}
