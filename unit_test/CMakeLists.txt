configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_contact.json
  ${CMAKE_CURRENT_BINARY_DIR}/hertzian_contact.json
  COPYONLY
)
##--------------------------------------------------------------------------##
## On-node tests
##--------------------------------------------------------------------------##
macro(CabanaPD_add_tests)
  cmake_parse_arguments(CABANAPD_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(CABANAPD_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  if(CABANAPD_UNIT_TEST_MPI)
    list(APPEND CABANAPD_UNIT_TEST_MPIEXEC_NUMPROCS 2)
    if(MPIEXEC_MAX_NUMPROCS GREATER 2)
      list(APPEND CABANAPD_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
    endif()
  endif()
  set(CABANAPD_UNIT_TEST_NUMTHREADS 1 2)

  set(CABANAPD_UNIT_TEST_MAIN mpi_unit_test_main.cpp)

  foreach(_device SERIAL PTHREAD OPENMP CUDA HIP)
    if(Kokkos_ENABLE_${_device})
      string(TOUPPER ${_device} _uppercase_device)
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_uppercase_device})
      file(MAKE_DIRECTORY ${_dir})
      foreach(_test ${CABANAPD_UNIT_TEST_NAMES})
        set(_file ${_dir}/tst${_test}_${_uppercase_device}.cpp)
        file(WRITE ${_file}
          "#include <Test${_uppercase_device}_Category.hpp>\n"
          "#include <tst${_test}.hpp>\n"
          )
        set(_target ${_test}_test_${_uppercase_device})
        add_executable(${_target} ${_file} ${CABANAPD_UNIT_TEST_MAIN})
        target_include_directories(${_target} PRIVATE ${_dir} ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${_target} PRIVATE CabanaPD ${gtest_target})

        foreach(_np ${CABANAPD_UNIT_TEST_MPIEXEC_NUMPROCS})
          if(_device STREQUAL PTHREAD OR _device STREQUAL OPENMP)
            foreach(_thread ${CABANAPD_UNIT_TEST_NUMTHREADS})
              add_test(NAME ${_target}_NP_${_np}_NT_${_thread} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
                ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=${_thread})
            endforeach()
          else()
            add_test(NAME ${_target}_NP_${_np} COMMAND
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_POSTFLAGsS} ${gtest_args})
          endif()
        endforeach()
      endforeach()
    endif()
  endforeach()
endmacro()

CabanaPD_add_tests(NAMES Particles Force Integrator Hertz)

CabanaPD_add_tests(MPI NAMES Comm)
