# =============================================================================
# Unit tests of ParMooN.
#
# Every test is a single source file test/<name>.c++ with its own main(). A test
# is compiled against the 2D library, the 3D library, or both (then two
# executables <name>_2d and <name>_3d are created). The 2D library is not built
# in the MPI configuration, so 2D tests exist only for the sequential/OpenMP
# builds. Tests are registered with ctest; 'make check' builds and runs them.
#
# This file was reconstructed after the original one had been lost; the
# assignment of tests to space dimensions was determined by compiling each test
# against both configurations.
# =============================================================================

set(parmoon_tests "")

# add one test executable linked against the library of dimension 'dim'
# (2 or 3). The executable is named <name>_<dim>d, the ctest entry likewise.
macro(add_parmoon_test name dim)
  set(_target ${name}_${dim}d)
  add_executable(${_target} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.c++")
  set_target_properties(${_target} PROPERTIES
                        OUTPUT_NAME ${_target}_${PARMOON_PARALLEL_TYPE})
  target_link_libraries(${_target} PUBLIC
                        parmoon_${dim}d_${PARMOON_PARALLEL_TYPE})
  if(PARMOON_USING_MPI)
    add_test(NAME ${_target}
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:${_target}>)
  else()
    add_test(NAME ${_target} COMMAND $<TARGET_FILE:${_target}>)
  endif()
  list(APPEND parmoon_tests ${_target})
endmacro(add_parmoon_test)

# tests which compile with both libraries
macro(add_parmoon_test_2d_3d name)
  if(NOT PARMOON_USING_MPI)
    add_parmoon_test(${name} 2)
  endif()
  add_parmoon_test(${name} 3)
endmacro(add_parmoon_test_2d_3d)

# tests which only compile with the 2D library (sequential/OpenMP builds only)
macro(add_parmoon_test_2d name)
  if(NOT PARMOON_USING_MPI)
    add_parmoon_test(${name} 2)
  endif()
endmacro(add_parmoon_test_2d)

# tests which only compile with the 3D library
macro(add_parmoon_test_3d name)
  add_parmoon_test(${name} 3)
endmacro(add_parmoon_test_3d)

# dimension independent tests: built once, against the 2D library where it
# exists (sequential/OpenMP builds), otherwise against the 3D library
macro(add_parmoon_test_any name)
  if(PARMOON_USING_MPI)
    add_parmoon_test(${name} 3)
  else()
    add_parmoon_test(${name} 2)
  endif()
endmacro(add_parmoon_test_any)

###############################################################################
# dimension independent tests (linear algebra, database, ...); built once
add_parmoon_test_any(block_matrix_test)
add_parmoon_test_any(block_vector_test)
add_parmoon_test_any(dense_matrix_test)
add_parmoon_test_any(direct_solver_test)
add_parmoon_test_any(iterative_solver_test)
add_parmoon_test_any(matrix_test)
add_parmoon_test_any(parameter_database_test)

###############################################################################
# tests for both 2D and 3D
add_parmoon_test_2d_3d(Hdiv_element_test)
add_parmoon_test_2d_3d(adaptive_hanging_test)
add_parmoon_test_2d_3d(analytical_function_test)
add_parmoon_test_2d_3d(boundary_condition_test)
add_parmoon_test_2d_3d(boundary_data_test)
add_parmoon_test_2d_3d(cd_dg_test)
add_parmoon_test_2d_3d(collection_test)
add_parmoon_test_2d_3d(darcy_test)
add_parmoon_test_2d_3d(derived_system_class_test)
add_parmoon_test_2d_3d(fe_function_test)
add_parmoon_test_2d_3d(function_evaluation_test)
add_parmoon_test_2d_3d(linesEval_test)
add_parmoon_test_2d_3d(multigrid_test)
add_parmoon_test_2d_3d(quadrature_test)
add_parmoon_test_2d_3d(read_write_fe_function_test)
add_parmoon_test_2d_3d(scott_vogelius_test)
add_parmoon_test_2d_3d(stokes_equal_order_test)
add_parmoon_test_2d_3d(structure_test)

###############################################################################
# 2D only
add_parmoon_test_2d(block_fe_matrix_test)
add_parmoon_test_2d(cd2d_adaptive_test)
add_parmoon_test_2d(cd2d_afc_adaptive_test)
add_parmoon_test_2d(cd2d_afc_test)
add_parmoon_test_2d(cd2d_slope_limiter_test)
add_parmoon_test_2d(cd2d_test)
add_parmoon_test_2d(lsc_test)
add_parmoon_test_2d(nse2d_dg_test)
add_parmoon_test_2d(nse2d_supg_test)
add_parmoon_test_2d(nse2d_test)
add_parmoon_test_2d(simplified_systems_test)
add_parmoon_test_2d(slope_limiter_test)
add_parmoon_test_2d(snaps_pod_rom_test)
add_parmoon_test_2d(tcd2d_femfct_test)
add_parmoon_test_2d(tcd2d_test)
add_parmoon_test_2d(tnse2d_rbvms_test)
add_parmoon_test_2d(tnse2d_test)
# brinkman2d_test.c++ includes "Brinkman2D.h", which no longer exists in this
# repository; it is therefore not built.
# add_parmoon_test_2d(brinkman2d_test)

###############################################################################
# 3D only
add_parmoon_test_3d(cd3d_afc_test)
add_parmoon_test_3d(cd3d_solvers_test)
add_parmoon_test_3d(mesh3d_test)
add_parmoon_test_3d(nse3d_solvers_test)
add_parmoon_test_3d(sandwich_test)
add_parmoon_test_3d(tcd3d_femfct_test)
add_parmoon_test_3d(tcd3d_solvers_test)
add_parmoon_test_3d(tnse3d_test)

###############################################################################
# tests requiring MPI (and MUMPS)
if(PARMOON_USING_MPI)
  add_parmoon_test_3d(domain_partitioning_read_write_test)
  if(PARMOON_WITH_MUMPS)
    add_parmoon_test_3d(mumps_test_1)
    add_parmoon_test_3d(mumps_test_2)
    add_parmoon_test_3d(mumps_wrapper_test)
  endif()
endif()

# make the list of test targets available to the parent (for 'make check')
set(parmoon_tests ${parmoon_tests} PARENT_SCOPE)
