include(${PROJECT_SOURCE_DIR}/test_harness/CabanaPD_testing_macros.cmake)

#Regression tests without MPI
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_contact.json ${CMAKE_CURRENT_BINARY_DIR}/hertzian_contact.json COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_jkr_contact.json ${CMAKE_CURRENT_BINARY_DIR}/hertzian_jkr_contact.json COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/mechanical_cube_single_material.json
  ${CMAKE_CURRENT_BINARY_DIR}/mechanical_cube_single_material.json
  COPYONLY
)

CabanaPD_add_tests(PREFIX RegressionTest NAMES Hertz HertzJKR MechanicalCubeSingleMaterial)

#Regression tests with MPI
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputs/elastic_wave.json ${CMAKE_CURRENT_BINARY_DIR}/elastic_wave.json COPYONLY )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputs/kalthoff_winkler.json ${CMAKE_CURRENT_BINARY_DIR}/kalthoff_winkler.json COPYONLY )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputs/crack_inclusion.json ${CMAKE_CURRENT_BINARY_DIR}/crack_inclusion.json COPYONLY )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/cube_linear_temperature.json
  ${CMAKE_CURRENT_BINARY_DIR}/cube_linear_temperature.json
  COPYONLY
)

CabanaPD_add_tests(MPI PREFIX RegressionTest NAMES ElasticWave KalthoffWinkler CrackInclusion CubeLinearTemperature)
