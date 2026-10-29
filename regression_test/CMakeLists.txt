configure_file(inputs/hertzian_contact.json hertzian_contact.json COPYONLY)
configure_file(inputs/hertzian_jkr_contact.json hertzian_jkr_contact.json COPYONLY)

include(${PROJECT_SOURCE_DIR}/test_harness/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(PREFIX RegressionTest NAMES Hertz HertzJKR)

configure_file(inputs/elastic_wave.json elastic_wave.json COPYONLY )
configure_file(inputs/kalthoff_winkler.json kalthoff_winkler.json COPYONLY )
configure_file(inputs/crack_inclusion.json crack_inclusion.json COPYONLY )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/cube_linear_temperature.json
  ${CMAKE_CURRENT_BINARY_DIR}/cube_linear_temperature.json
  COPYONLY
)

CabanaPD_add_tests(MPI PREFIX RegressionTest NAMES ElasticWave KalthoffWinkler CrackInclusion CubeLinearTemperature)
