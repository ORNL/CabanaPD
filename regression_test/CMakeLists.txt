configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_contact.json
  ${CMAKE_CURRENT_BINARY_DIR}/hertzian_contact.json
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_jkr_contact.json
  ${CMAKE_CURRENT_BINARY_DIR}/hertzian_jkr_contact.json
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/cube_single_material_pmb.json
  ${CMAKE_CURRENT_BINARY_DIR}/cube_single_material_pmb.json
  COPYONLY
)

include(${PROJECT_SOURCE_DIR}/test_harness/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(PREFIX RegressionTest NAMES Hertz HertzJKR CubeSingleMaterial)
