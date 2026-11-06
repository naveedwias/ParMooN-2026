#include <Domain.h>
#include <Database.h>
#include "ConvectionDiffusion.h"
#include "ParMooN.h"
#include <Chrono.h>

int main(int argc, char* argv[])
{
  auto parmoon_db = parmoon::parmoon_initialize(argc, argv);
  
  TDomain domain(parmoon_db);
  // refine grid
  domain.refine_and_get_hierarchy_of_collections(parmoon_db);
  
  // write grid into an Postscript file
  if(parmoon_db["output_write_ps"])
    domain.PS("Domain.ps", It_Finest, 0);
   
  //=========================================================================
  ConvectionDiffusion<2> cd2d(domain, parmoon_db);
  
  Chrono timer;
  
  cd2d.assemble();
  cd2d.solve();
  
  timer.print_total_time("whole solution");
  
  cd2d.output();
  //=========================================================================
  parmoon::parmoon_finalize();
}
