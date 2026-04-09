configure_file(inputs/hertzian_contact.json hertzian_contact.json COPYONLY)
configure_file(inputs/hertzian_jkr_contact.json hertzian_jkr_contact.json COPYONLY)

include(${PROJECT_SOURCE_DIR}/test_harness/CabanaPD_testing_macros.cmake)

CabanaPD_add_tests(PREFIX RegressionTest NAMES Hertz HertzJKR)

configure_file(inputs/crack_inclusion.json crack_inclusion.json COPYONLY )

CabanaPD_add_tests(MPI PREFIX RegressionTest NAMES CrackInclusion)
