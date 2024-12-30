add_executable(PowderFill powder_fill.cpp)
target_link_libraries(PowderFill LINK_PUBLIC CabanaPD)
install(TARGETS PowderFill DESTINATION ${CMAKE_INSTALL_BINDIR})
