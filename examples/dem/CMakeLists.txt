add_executable(PowderFill powder_fill.cpp)
target_link_libraries(PowderFill LINK_PUBLIC CabanaPD)
install(TARGETS PowderFill DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(AoR angle_repose.cpp)
target_link_libraries(AoR LINK_PUBLIC CabanaPD)
install(TARGETS AoR DESTINATION ${CMAKE_INSTALL_BINDIR})
