// First order Raviart-Thomas vector element, nonconforming, 2D

static double NF_N_T_RT1_CB_2D_T[3] = {-std::sqrt(3./5.), 0, std::sqrt(3./5.)};

// three point Gauss quadrature for edge dofs, seven-point formula for inner 
// dofs
static const double NF_N_T_RT1_CB_2D_eq[3] = 
 { 0.5*NF_N_T_RT1_CB_2D_T[0]+0.5,  0.5*NF_N_T_RT1_CB_2D_T[1]+0.5,
   0.5*NF_N_T_RT1_CB_2D_T[2]+0.5 };
static const double NF_N_T_RT1_CB_2D_ew[3] = { 5./18., 8./18., 5./18. };

static double NF_N_T_RT1_CB_2D_Xi[16] = 
 { NF_N_T_RT1_CB_2D_eq[0], NF_N_T_RT1_CB_2D_eq[1], NF_N_T_RT1_CB_2D_eq[2],
   NF_N_T_RT1_CB_2D_eq[2], NF_N_T_RT1_CB_2D_eq[1], NF_N_T_RT1_CB_2D_eq[0],
   0, 0, 0,
   0.333333333333333333333333333333333,
   0.797426985353087322398025276169754,
   0.101286507323456338800987361915123,
   0.101286507323456338800987361915123,
   0.059715871789769820459117580973106,
   0.470142064105115089770441209513447, 
   0.470142064105115089770441209513447 };
static double NF_N_T_RT1_CB_2D_Eta[16] = 
 { 0, 0, 0,
   NF_N_T_RT1_CB_2D_eq[0], NF_N_T_RT1_CB_2D_eq[1], NF_N_T_RT1_CB_2D_eq[2],
   NF_N_T_RT1_CB_2D_eq[2], NF_N_T_RT1_CB_2D_eq[1], NF_N_T_RT1_CB_2D_eq[0],
   0.333333333333333333333333333333333, 
   0.101286507323456338800987361915123,
   0.797426985353087322398025276169754, 
   0.101286507323456338800987361915123, 
   0.470142064105115089770441209513447,
   0.059715871789769820459117580973106,
   0.470142064105115089770441209513447 };

   // inner weights
static const double NF_N_T_RT1_CB_2D_iw[7] = 
 { 0.1125, 
   0.0629695902724135762978419727500906,
   0.0629695902724135762978419727500906,
   0.0629695902724135762978419727500906,
   0.0661970763942530903688246939165759,
   0.0661970763942530903688246939165759,
   0.0661970763942530903688246939165759 };



void NF_N_T_RT1_CB_2D_EvalAll(const TCollection *, const TBaseCell *Cell,
                           const double *PointValues, double *Functionals)
{
  // short names
  const double * iw = NF_N_T_RT1_CB_2D_iw;
  
  // set all Functionals to zero first
  for(unsigned int i = 0; i < 8; ++i)
    Functionals[i] = 0;
  
  // on the reference triangle with points (0,0), (1,0), (0,1) 
  if(Cell == nullptr)
  {
    //inner dofs
    for(unsigned int i = 0; i < 7; ++i)
    {
      Functionals[0] += PointValues[9+i] *iw[i];
      Functionals[1] += PointValues[25+i]*iw[i];
    }
  }
  else // on a real cell
  {
    double x0, x1, x2, y0, y1, y2, z; // z remains zero in 2D
    Cell->GetVertex(0)->GetCoords(x0, y0, z);
    Cell->GetVertex(1)->GetCoords(x1, y1, z);
    Cell->GetVertex(2)->GetCoords(x2, y2, z);
    // length of edge, and outer normal
    
    // the measure of the cell multiplied by the inverse measure of the 
    // refernce cell
    double measure = 2*Cell->GetMeasure();
    
    TTriaAffin referenceTransform;
    referenceTransform.SetCell(Cell);
    // transform the gradient of the (scalar) function phi(xi,eta) = xi
    // its gradient is (1,0) which is the vector with which we multiply to get
    // the correct dof
    
    double uxi[2] = {  1,0 };
    double ueta[2] = { 0,1 };
    for(unsigned int d = 0; d < 2; ++d) // inner dofs
    {
      double uref = 0., uxiref = uxi[d], uetaref = ueta[d];
      double uorig, uxorig, uyorig;
      for(unsigned int i = 0; i < 7; ++i) // loop over inner quadrature points
      {
        double x = NF_N_T_RT1_CB_2D_Xi[9+i], y = NF_N_T_RT1_CB_2D_Eta[9+i];
        referenceTransform.GetOrigValues(x, y, 1, &uref, &uxiref, &uetaref, 
                                         &uorig, &uxorig, &uyorig);
        Functionals[d] += ( PointValues[9+i]*uxorig
                              +PointValues[25+i]*uyorig ) * iw[i];
      }
    }
    Functionals[0] *= measure;
    Functionals[1] *= measure;
  }
}

void NF_N_T_RT1_CB_2D_EvalEdge(const TCollection *, const TBaseCell *Cell, int Joint,
                            const double *PointValues, double *Functionals)
{
}
