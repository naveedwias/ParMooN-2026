// ======================================================================
// Example from V. John and P. Knobloch.
// On the performance of SOLD methods for convection-diffusion problems
// with interior layers. Int. J. Comput. Sci. Math., 1(2-4):245–258, 2007.
//
// Used with mesh UnitSquare18Triangles.GEO
// ======================================================================

#include "math.h"
#include "FEDatabase.h"


double DIFFUSION;


void ExampleFile()
{
  Output::print<1>("Example: TwoInteriorLayersRegularized.h");
}


auto& Exact = unknown_solution_2d;


void BoundCondition(int BdComp, double, BoundCond &cond)
{
  if(BdComp == 3)
	  cond = NEUMANN;
  else
	  cond = DIRICHLET;
}


void BoundValue(int BdComp, double Param, double &value)
{
  double paramXi = 1.0e-3;
  
  if(BdComp == 0)
  {
    if( (Param > 1.0/3.0 + paramXi) && (Param < 2.0/3.0 - paramXi) )
       value = 1.0;
    else if( (Param >= 1.0/3.0 - paramXi) && (Param <= 1.0/3.0 + paramXi) )
    {
      double argCos = M_PI * (1.0/3.0 + paramXi - Param) / (2.0*paramXi);
      double oneMinusCos = 1.0 - std::cos(argCos);
      
      value = 1.0 - oneMinusCos*oneMinusCos / 4.0;
    }   
    else if( (Param >= 2.0/3.0 - paramXi) && (Param <= 2.0/3.0 + paramXi) )
    {
      double argCos = M_PI * (-2.0/3.0 + paramXi + Param) / (2.0*paramXi);
	    double oneMinusCos = 1.0 - std::cos(argCos);
      
	    value = 1.0 - oneMinusCos*oneMinusCos / 4.0;
    }
    else
      value = 0;
  }
  else
    value = 0;
}


void BilinearCoeffs(int n_points, const double *X, const double *Y,
                    const double *const*, double **coeffs)
{
  double eps = DIFFUSION;
  for(int i = 0; i < n_points; i++)
  {
    double x = X[i];
    double y = Y[i];
    coeffs[i][0] = eps;
    coeffs[i][1] = -y;
    coeffs[i][2] = x;
    coeffs[i][3] = 0;
    coeffs[i][4] = 0;
  }
}


void ComputeBdLayer(const TFEFunction2D& ufct, double xcut, double yStart, int direction, double& endInDirection, double& layerWidth)
{
  // Consider a line starting at (xcut, yStart) going in the positive y-direction.
  // This should be behind the cylinder where the solution is one and then 
  // (going up) zero.
  // We will go along that line to find the first point where ufct is below 
  // valmax, then we go further along that line to find the first point where 
  // ufct is below valmin. Remembering where these two transitions appeared, we
  // are able to compute the "layer width". This is described also in 
  // Augustin, Caiazzo, Fiebach, Fuhrmann, John, Linke, Umla: "An assessment of 
  // discretizations for convection-dominated convection–diffusion equations",
  // 2011.
  // We avoid having to search all cells for every point, which would be very
  // expensive, by only searching the neighboring cells from where the previous
  // point was found.
  constexpr int bound_points = 50001;
  constexpr double valmin = 0.1;
  constexpr double valmax = 0.9;
  constexpr double Width = 0.5;

  if(std::abs(direction) != 1)
    ErrThrow("Direction must be 1 or -1.");
  
  double ystart = yStart;
  double yend = Width + yStart;
  double h = Width / (bound_points-1);
  parmoon::Point p(xcut, ystart);

  // First, we have to determine the cell in which the starting point (xcut, 0)
  // lies. Therefore, we have to do a loop over all cells and determine the
  // cell. This cell is called starting_cell_nr.
  auto collection = ufct.GetFESpace()->GetCollection();
  auto n_cells = collection->GetN_Cells();
  int starting_cell_nr = 0;
  for(int cell_i = 0; cell_i < n_cells; ++cell_i)
  {
    auto current_cell = collection->GetCell(cell_i);
    bool is_point_in_cell = current_cell->PointInCell(p);
    if (is_point_in_cell)
    {
      starting_cell_nr = cell_i;
      break;
    }
  }
  
  // Now a loop over all y values can be performed. In each iteration we have to
  // determine the cell in which the current point lies. For this purpose the
  // current cell number and the current cell are determined. In the beginning
  // the current cell number is of course the previously computed
  // starting_cell_nr.
  int current_cell_nr = starting_cell_nr;
  int jstart = 0;
  for(int j = 0; j < bound_points; j++)
  {
    p.y = yStart + direction * h * j;
    collection->find_cell(p, current_cell_nr);
    double val;
    auto current_cell = collection->GetCell(current_cell_nr);
    ufct.FindValueLocal(current_cell, current_cell_nr, p.x, p.y, &val);
    if(val < valmax)
    {
      ystart = p.y;
      jstart = j;
      break;
    }
  }

  for(int j=jstart+1;j<bound_points;j++)
  {
    p.y = yStart + direction * h * j;
    collection->find_cell(p, current_cell_nr);
    double val;
    auto current_cell = collection->GetCell(current_cell_nr);
    ufct.FindValueLocal(current_cell, current_cell_nr, p.x, p.y, &val);
    if(val < valmin)
    {
      yend = p.y;
      break;
    }
  }
  auto lw = std::abs(yend - ystart);

  if (lw >= Width)
  {
    Output::warn("TwoInteriorLayersRegularized::compute_layer_width", "Failed to compute layer ",
        "width at x=",xcut,". I did not find an end point.");
  }
  if (lw <= h)
  {
    Output::warn("TwoInteriorLayersRegularized::compute_layer_width", "Layer width equals step ",
        "width at x=",xcut,"!");
  }
  
  layerWidth = lw;
  endInDirection = yend;
 }
 
 
void postprocessing(ConvectionDiffusion<2> & cd2d)
{
  auto & uh = cd2d.get_function();
   
  double lwUpper, lwLower, endUpper, endLower;
  
  ComputeBdLayer(uh, 0.0, 0.5, 1, endUpper, lwUpper);
  ComputeBdLayer(uh, 0.0, 0.5, -1, endLower, lwLower);
  
  Output::print("Upper layer width at 0.0: ", std::setprecision(12), lwUpper);
  Output::print("Lower layer width at 0.0: ", std::setprecision(12), lwLower);
  
  double outflowProfileWidth = endUpper - endLower;
  
  Output::print("Output profile width: ", std::setprecision(12), outflowProfileWidth);
}
