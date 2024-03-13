cmake_minimum_required(VERSION 3.11)
project(CabanaPD LANGUAGES CXX C)
set(PROJECT_VERSION "0.2.0-dev")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##---------------------------------------------------------------------------##
# Set up main options (inherit from Kokkos and Cabana CMake)
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED)

macro(CabanaPD_check_optional)
  cmake_parse_arguments(CABANA "" "OPTION" "" ${ARGN})
  if( Cabana_ENABLE_${CABANA_OPTION} )
    message( STATUS "Cabana includes ${CABANA_OPTION}" )
  endif()
endmacro()

CabanaPD_check_optional( OPTION HDF5 )
CabanaPD_check_optional( OPTION SILO )

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()
find_package(nlohmann_json 3.10.0 QUIET)
if(NOT NLOHMANN_JSON_FOUND)
  include(FetchContent)
  # Using most recent release here
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CabanaPD_Config.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/CabanaPD_Config.cmake @ONLY)

##---------------------------------------------------------------------------##
## Print the Git revision number to stdout
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   CabanaPD_GIT_COMMIT_HASH
        )
ELSE()
    SET(CabanaPD_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "CabanaPD Revision = '${CabanaPD_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Build CabanaPD
##---------------------------------------------------------------------------##
add_subdirectory(src)
add_subdirectory(examples)

##---------------------------------------------------------------------------##
## Unit tests
##---------------------------------------------------------------------------##
option(CabanaPD_ENABLE_TESTING "Build tests" OFF)
if(CabanaPD_ENABLE_TESTING)
  find_package(GTest 1.10 REQUIRED)
  # Workaround for FindGTest module in CMake older than 3.20
  if(TARGET GTest::gtest)
    set(gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(gtest_target GTest::GTest)
  else()
    message(FATAL_ERROR "bug in GTest find module workaround")
  endif()
  enable_testing()
  add_subdirectory(unit_test)
endif()

##---------------------------------------------------------------------------##
## Clang format
##---------------------------------------------------------------------------##
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.[c,h]pp unit_test/*.[c,h]pp examples/*.[c,h]pp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
