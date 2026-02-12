include(${PROJECT_SOURCE_DIR}/cmake/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(PREFIX UnitTest NAMES Particles Force Integrator NormalRepulsion)

CabanaPD_add_tests(MPI PREFIX UnitTest NAMES Comm)
