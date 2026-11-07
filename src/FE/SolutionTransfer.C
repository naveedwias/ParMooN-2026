#ifdef __2D__
#include "SolutionTransfer.h"
#include "BaseCell.h"
#include "Collection.h"
#include "FEDatabase.h"
#include "FESpace2D.h"
#include "FiniteElement.h"
#include "MooNMD_Io.h"
#include "NodalFunctional.h"
#include "Vertex.h"
#include <cmath>

SolutionTransfer::PointKey SolutionTransfer::key(double x, double y)
{
  return {std::llround(x*1e9), std::llround(y*1e9)};
}

bool SolutionTransfer::point_in_triangle(const TBaseCell* c, double x, double y,
                                         double tol)
{
  double x0 = c->GetVertex(0)->GetX(), y0 = c->GetVertex(0)->GetY();
  double x1 = c->GetVertex(1)->GetX(), y1 = c->GetVertex(1)->GetY();
  double x2 = c->GetVertex(2)->GetX(), y2 = c->GetVertex(2)->GetY();
  double det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
  double l1 = ((x-x0)*(y2-y0) - (x2-x0)*(y-y0))/det;
  double l2 = ((x1-x0)*(y-y0) - (x-x0)*(y1-y0))/det;
  return l1 >= -tol && l2 >= -tol && l1 + l2 <= 1. + tol;
}

namespace
{
  // nodal points of the element of 'space' on cell 'cell_no', placed on an
  // arbitrary triangle (a, b, c) by the affine map
  void nodal_points(const std::shared_ptr<const TFESpace2D>& space, int cell_no,
                    const double* a, const double* b, const double* c,
                    std::vector<std::pair<double,double>>& pts)
  {
    const FiniteElement& element = space->get_fe(cell_no);
    const NodalFunctional* nf = element.GetNodalFunctional();
    int n;
    const double *xi, *eta;
    nf->GetPointsForAll(n, xi, eta);
    for(int j = 0; j < n; ++j)
      pts.emplace_back(a[0] + xi[j]*(b[0]-a[0]) + eta[j]*(c[0]-a[0]),
                       a[1] + xi[j]*(b[1]-a[1]) + eta[j]*(c[1]-a[1]));
  }

  // nodal points of the triangle v and of all its possible children
  void candidate_points(const std::shared_ptr<const TFESpace2D>& space,
                        int cell_no, const double v[3][2],
                        std::vector<std::pair<double,double>>& pts)
  {
    double m[3][2];
    for(int k = 0; k < 2; ++k)
    {
      m[0][k] = 0.5*(v[0][k]+v[1][k]);
      m[1][k] = 0.5*(v[1][k]+v[2][k]);
      m[2][k] = 0.5*(v[2][k]+v[0][k]);
    }
    nodal_points(space, cell_no, v[0], v[1], v[2], pts);
    nodal_points(space, cell_no, v[0], m[0], m[2], pts);
    nodal_points(space, cell_no, m[0], v[1], m[1], pts);
    nodal_points(space, cell_no, m[2], m[1], v[2], pts);
    nodal_points(space, cell_no, m[0], m[1], m[2], pts);
    nodal_points(space, cell_no, v[0], m[0], v[2], pts);
    nodal_points(space, cell_no, m[0], v[1], v[2], pts);
    nodal_points(space, cell_no, v[1], m[1], v[0], pts);
    nodal_points(space, cell_no, m[1], v[2], v[0], pts);
    nodal_points(space, cell_no, v[2], m[2], v[1], pts);
    nodal_points(space, cell_no, m[2], v[0], v[1], pts);
  }

  const TBaseCell* leaf_containing(const TBaseCell* c, double x, double y)
  {
    while(c != nullptr && c->ExistChildren())
    {
      const TBaseCell* next = nullptr;
      for(int i = 0; i < c->GetN_Children(); ++i)
        if(SolutionTransfer::point_in_triangle(c->GetChild(i), x, y))
        { next = c->GetChild(i); break; }
      c = next;
    }
    return c;
  }

  void vertices(const TBaseCell* cell, double v[3][2])
  {
    for(int k = 0; k < 3; ++k)
    {
      v[k][0] = cell->GetVertex(k)->GetX();
      v[k][1] = cell->GetVertex(k)->GetY();
    }
  }
}

SolutionTransfer::NodeValueMap SolutionTransfer::record(const TFEFunction2D& u)
{
  NodeValueMap m;
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  int n_cells = coll->GetN_Cells();
  std::map<const TBaseCell*, int> index;
  for(int i = 0; i < n_cells; ++i)
    index[coll->GetCell(i)] = i;
  std::vector<std::pair<double,double>> pts;
  double val[3], v[3][2];
  for(int i = 0; i < n_cells; ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    if(cell->GetN_Vertices() != 3)
      ErrThrow("SolutionTransfer: only triangles are supported");
    vertices(cell, v);
    pts.clear();
    candidate_points(space, i, v, pts);
    for(auto& p : pts)
    {
      PointKey k = key(p.first, p.second);
      if(m.count(k)) continue;
      u.FindValueLocal(cell, i, p.first, p.second, val);
      m[k] = val[0];
    }
    const TBaseCell* parent = cell->GetParent();
    if(parent != nullptr && parent->GetChild(0) == cell)
    {
      vertices(parent, v);
      pts.clear();
      candidate_points(space, i, v, pts);
      for(auto& p : pts)
      {
        PointKey k = key(p.first, p.second);
        if(m.count(k)) continue;
        const TBaseCell* leaf = leaf_containing(parent, p.first, p.second);
        auto it = (leaf != nullptr) ? index.find(leaf) : index.end();
        if(it == index.end()) continue;
        u.FindValueLocal(leaf, it->second, p.first, p.second, val);
        m[k] = val[0];
      }
    }
  }
  return m;
}

std::vector<SolutionTransfer::NodeValueMap>
SolutionTransfer::record(const TFEVectFunct2D& u)
{
  std::vector<NodeValueMap> maps;
  for(int c = 0; c < u.GetN_Components(); ++c)
    maps.push_back(record(*u.GetComponent(c)));
  return maps;
}

void SolutionTransfer::apply(TFEFunction2D& u, const NodeValueMap& values)
{
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  double* entries = u.GetValues();
  std::vector<std::pair<double,double>> pts;
  double v[3][2];
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    vertices(coll->GetCell(i), v);
    pts.clear();
    nodal_points(space, i, v[0], v[1], v[2], pts);
    const int* DOF = space->GetGlobalDOF(i);
    for(size_t j = 0; j < pts.size(); ++j)
    {
      auto it = values.find(key(pts[j].first, pts[j].second));
      if(it == values.end())
        ErrThrow("SolutionTransfer: no recorded value at (", pts[j].first, ",",
                 pts[j].second, ")");
      entries[DOF[j]] = it->second;
    }
  }
}

void SolutionTransfer::apply(TFEVectFunct2D& u, const std::vector<NodeValueMap>& values)
{
  if((int) values.size() != u.GetN_Components())
    ErrThrow("SolutionTransfer: wrong number of components");
  for(int c = 0; c < u.GetN_Components(); ++c)
  {
    auto comp = u.GetComponent(c);
    apply(*comp, values[c]);
  }
}
#endif // __2D__
