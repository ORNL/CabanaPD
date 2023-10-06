configure_file(CabanaPD_config.hpp.cmakein CabanaPD_config.hpp @ONLY)

file(GLOB HEADERS GLOB *.hpp)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaPD_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(CabanaPD INTERFACE)

target_include_directories(CabanaPD INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(CabanaPD INTERFACE Cabana::cabanacore Cabana::Cajita)

install(TARGETS CabanaPD DESTINATION ${CMAKE_INSTALL_LIBDIR})
