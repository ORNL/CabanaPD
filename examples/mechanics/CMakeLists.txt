add_executable(ElasticWave elastic_wave.cpp)
target_link_libraries(ElasticWave LINK_PUBLIC CabanaPD)

add_executable(KalthoffWinkler kalthoff_winkler.cpp)
target_link_libraries(KalthoffWinkler LINK_PUBLIC CabanaPD)

add_executable(CrackBranching crack_branching.cpp)
target_link_libraries(CrackBranching LINK_PUBLIC CabanaPD)

add_executable(FragmentingCylinder fragmenting_cylinder.cpp)
target_link_libraries(FragmentingCylinder LINK_PUBLIC CabanaPD)

#add_executable(DiskImpact disk_impact.cpp)
#target_link_libraries(DiskImpact LINK_PUBLIC CabanaPD)

#install(TARGETS ElasticWave KalthoffWinkler CrackBranching FragmentingCylinder DiskImpact DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ElasticWave KalthoffWinkler CrackBranching FragmentingCylinder DESTINATION ${CMAKE_INSTALL_BINDIR})
