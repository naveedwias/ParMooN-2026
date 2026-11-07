// =============================================================================
// Transfer of Lagrange finite element functions between two meshes of an
// adaptive loop (refinement, coarsening, conforming closure).
//
// Before the mesh is changed, record() evaluates the function at every nodal
// point the next mesh can have: the nodal points of the cell itself, of its
// regularly refined children, of the children of all three bisections, and
// the same for the parent cell (the conforming closure rebuilds closure cells
// from their parent). The values are stored by (rounded) coordinates. After
// the mesh has been changed, apply() fills a function on the new mesh from
// the stored values. Coarsened cells only need values at nodes which were
// nodes of their children, so no point search on the old mesh is needed after
// it has been changed (its cells may already be deleted).
//
// Works for continuous and discontinuous Lagrange elements of any order on
// affine triangles; the nodal points must be point evaluations (Lagrange
// nodal functionals).
// =============================================================================
#ifndef SOLUTIONTRANSFER_H
#define SOLUTIONTRANSFER_H

#include "FEFunction2D.h"
#include "FEVectFunct2D.h"
#include <map>
#include <utility>
#include <vector>

class SolutionTransfer
{
  public:
    typedef std::pair<long long, long long> PointKey;
    typedef std::map<PointKey, double> NodeValueMap;

    /// @brief rounded coordinates used as key (10^-9 resolution)
    static PointKey key(double x, double y);

    /// @brief record the values of a scalar function (call before the mesh
    /// is changed); returns the map
    static NodeValueMap record(const TFEFunction2D& u);

    /// @brief record all components of a vector function
    static std::vector<NodeValueMap> record(const TFEVectFunct2D& u);

    /// @brief fill a function on the new mesh from recorded values; throws if
    /// a nodal point of the new mesh has no recorded value
    static void apply(TFEFunction2D& u, const NodeValueMap& values);
    static void apply(TFEVectFunct2D& u, const std::vector<NodeValueMap>& values);

    /// @brief point in (affine) triangle test by barycentric coordinates
    static bool point_in_triangle(const TBaseCell* cell, double x, double y,
                                  double tol = 1e-9);
};

#endif // SOLUTIONTRANSFER_H
