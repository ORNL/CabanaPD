include(${PROJECT_SOURCE_DIR}/cmake/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(NAMES Particles Force Integrator NormalRepulsion)

CabanaPD_add_tests(MPI NAMES Comm)
