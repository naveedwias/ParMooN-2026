#ifndef INCLUDE_GENERAL_PARMOON_H_
#define INCLUDE_GENERAL_PARMOON_H_

#include "ParameterDatabase.h"

namespace parmoon
{
  /**
   * @brief Initialize ParMooN. Call this at the beginning of any main function.
   *
   * This generates global objects which needs to be done exactly once. Also
   * this will read the input file, if present, and return the respective
   * ParameterDatabase.
   *
   * Note that a main program can be well formed even without this, but you have
   * to know what you're doing.
   */
  ParameterDatabase parmoon_initialize(int argc = 0, char* argv[] = nullptr);

  /**
   * @brief Finalize ParMooN. This does all the necessary internal cleanup for a
   * program started with parmoon_initialize.
   *
   * As with parmoon_initialize, you can do the cleanup yourself, but shouldn't.
   */
  void parmoon_finalize();

  /**
   * @brief In the MPI case, returns true on the root process (rank 0) only.
   * Otherwise, always returns true.
   */
  bool i_am_root();
}

#endif // INCLUDE_GENERAL_PARMOON_H_
