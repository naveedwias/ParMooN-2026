#ifndef __SIMPLIFIED_INFO_H__
#define __SIMPLIFIED_INFO_H__

#include <ParameterDatabase.h>

namespace SimplifiedInfo
{
  ParameterDatabase default_solver_db();

  ParameterDatabase default_saddle_solver_db(bool direct_u = false, bool direct_p = true);

  ParameterDatabase default_direct_solver_db();
}

#endif