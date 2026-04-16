include(${PROJECT_SOURCE_DIR}/test_harness/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(PREFIX UnitTest NAMES Particles Force Integrator NormalRepulsion Indexing)

CabanaPD_add_tests(MPI PREFIX UnitTest NAMES Comm)
