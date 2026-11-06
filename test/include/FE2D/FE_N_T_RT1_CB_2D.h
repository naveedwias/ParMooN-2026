// ***********************************************************************
// P1 Cell Bubbles for Raviart-Thomas vector element, nonconforming , 2D
// ***********************************************************************

// number of degrees of freedom
static int N_T_RT1_2D_CB_NDOF = 2;

// number of dofs on the closure of the joints
static int N_T_RT1_2D_CB_JointDOF = 0;

// number of inner dofs
static int N_T_RT1_2D_CB_NInner = 2;

// array containing the numbers for the inner dofs
static int N_T_RT1_2D_CB_Inner[2] = {0,1};

// number of outer dofs (dofs on edges)
static int N_T_RT1_2D_CB_NOuter = 0;
