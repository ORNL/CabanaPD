add_executable(cbnPD CabanaPD_main.cpp)
target_link_libraries(cbnPD LINK_PUBLIC CabanaPD)
install(TARGETS cbnPD DESTINATION examples)
