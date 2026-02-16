configure_file(CabanaPD_config.hpp.cmakein CabanaPD_config.hpp @ONLY)

file(GLOB HEADERS GLOB *.hpp)
file(GLOB FORCE_HEADERS GLOB force/*.hpp)
file(GLOB FORCE_MODEL_HEADERS GLOB force_models/*.hpp)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${FORCE_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/force)
install(FILES ${FORCE_MODEL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/force_models)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaPD_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(CabanaPD INTERFACE)
add_library(CabanaPD::CabanaPD ALIAS CabanaPD)

target_include_directories(CabanaPD INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(CabanaPD INTERFACE Cabana::Core Cabana::Grid nlohmann_json::nlohmann_json)

install(TARGETS CabanaPD
        EXPORT CabanaPD_Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT CabanaPD_Targets
  FILE CabanaPD_Targets.cmake
  NAMESPACE CabanaPD::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/CabanaPD)
