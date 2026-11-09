#ifdef __2D__
#include "SolutionTransfer.h"
#include "BaseCell.h"
#include "BaseFunctions.h"
#include "Collection.h"
#include "FESpace2D.h"
#include "FiniteElement.h"
#include "MooNMD_Io.h"
#include "NodalFunctional.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>

bool SolutionTransfer::point_in_triangle(const double* v, double x, double y,
                                         double tol)
{
  double det = (v[2]-v[0])*(v[5]-v[1]) - (v[4]-v[0])*(v[3]-v[1]);
  double l1 = ((x-v[0])*(v[5]-v[1]) - (v[4]-v[0])*(y-v[1]))/det;
  double l2 = ((v[2]-v[0])*(y-v[1]) - (x-v[0])*(v[3]-v[1]))/det;
  return l1 >= -tol && l2 >= -tol && l1 + l2 <= 1. + tol;
}

int SolutionTransfer::Snapshot::bucket_index(double x, double y) const
{
  int i = (int) std::floor((x - x_min)/dx);
  int j = (int) std::floor((y - y_min)/dy);
  i = std::min(std::max(i, 0), nx - 1);
  j = std::min(std::max(j, 0), ny - 1);
  return j*nx + i;
}

int SolutionTransfer::Snapshot::find_cell(double x, double y) const
{
  // the bucket of the point, then its neighbours in growing rings: a cell is
  // registered in every bucket its bounding box touches, so the first ring is
  // enough unless the point is outside the stored mesh (round-off at the
  // boundary) -- then the nearest cell centre is used
  int i0 = std::min(std::max((int) std::floor((x - x_min)/dx), 0), nx - 1);
  int j0 = std::min(std::max((int) std::floor((y - y_min)/dy), 0), ny - 1);
  for(int ring = 0; ring <= 2; ++ring)
  {
    for(int j = std::max(j0-ring, 0); j <= std::min(j0+ring, ny-1); ++j)
      for(int i = std::max(i0-ring, 0); i <= std::min(i0+ring, nx-1); ++i)
      {
        if(ring > 0 && std::abs(i-i0) < ring && std::abs(j-j0) < ring)
          continue; // already looked at
        for(int c : buckets[j*nx + i])
          if(point_in_triangle(vertices[c].data(), x, y))
            return c;
      }
  }
  // fall back: nearest cell centre
  int best = -1;
  double best_d = 1e300;
  for(size_t c = 0; c < vertices.size(); ++c)
  {
    const auto& v = vertices[c];
    double cx = (v[0]+v[2]+v[4])/3., cy = (v[1]+v[3]+v[5])/3.;
    double d = (cx-x)*(cx-x) + (cy-y)*(cy-y);
    if(d < best_d)
    { best_d = d; best = (int) c; }
  }
  return best;
}

double SolutionTransfer::Snapshot::evaluate(double x, double y) const
{
  int c = find_cell(x, y);
  if(c < 0)
    ErrThrow("SolutionTransfer: empty snapshot");
  const auto& v = vertices[c];
  // reference coordinates of the affine map x = v0 + xi (v1-v0) + eta (v2-v0)
  double det = (v[2]-v[0])*(v[5]-v[1]) - (v[4]-v[0])*(v[3]-v[1]);
  double xi  = ((x-v[0])*(v[5]-v[1]) - (v[4]-v[0])*(y-v[1]))/det;
  double eta = ((v[2]-v[0])*(y-v[1]) - (x-v[0])*(v[3]-v[1]))/det;
  int n = basis[c]->GetDimension();
  std::vector<double> phi(n);
  basis[c]->GetDerivatives(MultiIndex2D::D00, xi, eta, phi.data());
  double value = 0.;
  for(int k = 0; k < n; ++k)
    value += phi[k]*values[c][k];
  return value;
}

namespace
{
  // nodal points of the element of 'space' on cell 'cell_no' in physical
  // coordinates (Lagrange elements: the nodal functionals are point
  // evaluations, in the order of the local degrees of freedom)
  void nodal_points(const std::shared_ptr<const TFESpace2D>& space, int cell_no,
                    const double* v, std::vector<std::pair<double,double>>& pts)
  {
    const FiniteElement& element = space->get_fe(cell_no);
    int n;
    const double *xi, *eta;
    element.GetNodalFunctional()->GetPointsForAll(n, xi, eta);
    pts.clear();
    for(int j = 0; j < n; ++j)
      pts.emplace_back(v[0] + xi[j]*(v[2]-v[0]) + eta[j]*(v[4]-v[0]),
                       v[1] + xi[j]*(v[3]-v[1]) + eta[j]*(v[5]-v[1]));
  }
}

SolutionTransfer::Snapshot SolutionTransfer::record(const TFEFunction2D& u)
{
  Snapshot s;
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  int n_cells = coll->GetN_Cells();
  const double* entries = u.GetValues();
  s.vertices.resize(n_cells);
  s.basis.resize(n_cells);
  s.values.resize(n_cells);
  double x_max = -1e300, y_max = -1e300;
  s.x_min = s.y_min = 1e300;
  for(int i = 0; i < n_cells; ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    if(cell->GetN_Vertices() != 3)
      ErrThrow("SolutionTransfer: only triangles are supported");
    auto& v = s.vertices[i];
    for(int k = 0; k < 3; ++k)
    {
      v[2*k]   = cell->GetVertex(k)->GetX();
      v[2*k+1] = cell->GetVertex(k)->GetY();
      s.x_min = std::min(s.x_min, v[2*k]);
      x_max = std::max(x_max, v[2*k]);
      s.y_min = std::min(s.y_min, v[2*k+1]);
      y_max = std::max(y_max, v[2*k+1]);
    }
    const FiniteElement& element = space->get_fe(i);
    s.basis[i] = element.GetBaseFunct();
    int n_dof = element.GetN_DOF();
    const int* DOF = space->GetGlobalDOF(i);
    s.values[i].resize(n_dof);
    for(int k = 0; k < n_dof; ++k)
      s.values[i][k] = entries[DOF[k]];
  }
  // bucket grid: about one bucket per cell
  int n = std::max(1, (int) std::sqrt((double) n_cells));
  s.nx = s.ny = n;
  s.dx = std::max((x_max - s.x_min)/n, 1e-14);
  s.dy = std::max((y_max - s.y_min)/n, 1e-14);
  s.buckets.assign(n*n, {});
  for(int i = 0; i < n_cells; ++i)
  {
    const auto& v = s.vertices[i];
    double bx0 = std::min({v[0], v[2], v[4]}), bx1 = std::max({v[0], v[2], v[4]});
    double by0 = std::min({v[1], v[3], v[5]}), by1 = std::max({v[1], v[3], v[5]});
    int i0 = s.bucket_index(bx0, by0) % s.nx, j0 = s.bucket_index(bx0, by0)/s.nx;
    int i1 = s.bucket_index(bx1, by1) % s.nx, j1 = s.bucket_index(bx1, by1)/s.nx;
    for(int j = j0; j <= j1; ++j)
      for(int i2 = i0; i2 <= i1; ++i2)
        s.buckets[j*s.nx + i2].push_back(i);
  }
  return s;
}

std::vector<SolutionTransfer::Snapshot>
SolutionTransfer::record(const TFEVectFunct2D& u)
{
  std::vector<Snapshot> snapshots;
  for(int c = 0; c < u.GetN_Components(); ++c)
    snapshots.push_back(record(*u.GetComponent(c)));
  return snapshots;
}

void SolutionTransfer::apply(TFEFunction2D& u, const Snapshot& snapshot)
{
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  double* entries = u.GetValues();
  std::vector<std::pair<double,double>> pts;
  double v[6];
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    for(int k = 0; k < 3; ++k)
    {
      v[2*k]   = cell->GetVertex(k)->GetX();
      v[2*k+1] = cell->GetVertex(k)->GetY();
    }
    nodal_points(space, i, v, pts);
    const int* DOF = space->GetGlobalDOF(i);
    for(size_t j = 0; j < pts.size(); ++j)
      entries[DOF[j]] = snapshot.evaluate(pts[j].first, pts[j].second);
  }
}

void SolutionTransfer::apply(TFEVectFunct2D& u,
                             const std::vector<Snapshot>& snapshots)
{
  if((int) snapshots.size() != u.GetN_Components())
    ErrThrow("SolutionTransfer: wrong number of components");
  for(int c = 0; c < u.GetN_Components(); ++c)
  {
    auto comp = u.GetComponent(c);
    apply(*comp, snapshots[c]);
  }
}
#endif // __2D__
