add_executable(ElasticWave elastic_wave.cpp)
target_link_libraries(ElasticWave LINK_PUBLIC CabanaPD)

add_executable(KalthoffWinkler kalthoff_winkler.cpp)
target_link_libraries(KalthoffWinkler LINK_PUBLIC CabanaPD)

add_executable(CrackBranching crack_branching.cpp)
target_link_libraries(CrackBranching LINK_PUBLIC CabanaPD)

add_executable(ReadFile read_file.cpp)
target_link_libraries(ReadFile LINK_PUBLIC CabanaPD)

install(TARGETS ElasticWave KalthoffWinkler CrackBranching DESTINATION ${CMAKE_INSTALL_BINDIR})
