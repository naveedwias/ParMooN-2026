// =============================================================================
// Transfer of Lagrange finite element functions between two meshes of an
// adaptive loop (refinement, coarsening and conforming closure).
//
//   record()  takes a snapshot of a function BEFORE the mesh is changed: the
//             geometry of every cell of the current mesh and the local values
//             of the function on it. The snapshot is self-contained, so the
//             cells of the old mesh may be deleted afterwards (they are, for
//             the parts of the mesh which are coarsened).
//   apply()   fills a function on the NEW mesh: every nodal point of the new
//             mesh is located in the old mesh and the local polynomial of the
//             old cell is evaluated there (a uniform bucket grid makes the
//             point location O(1) per point).
//
// This is a true mesh-to-mesh interpolation and does not assume any relation
// between the two meshes, in particular no match of nodal points, so it
// survives a conforming closure which bisects a cell more than once. For
// Lagrange elements on affine triangles (the nodal functionals must be point
// evaluations) the result is the nodal interpolant of the old function; on
// cells which are not refined it reproduces the function exactly.
// =============================================================================
#ifndef SOLUTIONTRANSFER_H
#define SOLUTIONTRANSFER_H

#include "FEFunction2D.h"
#include "FEVectFunct2D.h"
#include <array>
#include <memory>
#include <vector>

class BaseFunctions;

class SolutionTransfer
{
  public:
    /// @brief a self-contained copy of a scalar FE function on a mesh
    class Snapshot
    {
      public:
        /// @brief value at the point (x, y); the point is located in the
        /// stored mesh (the nearest cell is used if it is outside)
        double evaluate(double x, double y) const;

        std::vector<std::array<double, 6>> vertices; //< x0 y0 x1 y1 x2 y2
        std::vector<const BaseFunctions*> basis;     //< per cell
        std::vector<std::vector<double>> values;     //< local values per cell
        // uniform bucket grid over the bounding box
        double x_min, y_min, dx, dy;
        int nx, ny;
        std::vector<std::vector<int>> buckets;

        int find_cell(double x, double y) const;
        int bucket_index(double x, double y) const;
    };

    /// @brief snapshot of a scalar function (call before the mesh is changed)
    static Snapshot record(const TFEFunction2D& u);
    /// @brief one snapshot per component of a vector function
    static std::vector<Snapshot> record(const TFEVectFunct2D& u);

    /// @brief fill a function on the new mesh from a snapshot
    static void apply(TFEFunction2D& u, const Snapshot& snapshot);
    static void apply(TFEVectFunct2D& u, const std::vector<Snapshot>& snapshots);

    /// @brief point in (affine) triangle test by barycentric coordinates
    static bool point_in_triangle(const double* v, double x, double y,
                                  double tol = 1e-10);
};

#endif // SOLUTIONTRANSFER_H
