add_executable(ElasticWave elastic_wave.cpp)
target_link_libraries(ElasticWave LINK_PUBLIC CabanaPD)
install(TARGETS ElasticWave DESTINATION examples)

add_executable(KalthoffWinkler kalthoff_winkler.cpp)
target_link_libraries(KalthoffWinkler LINK_PUBLIC CabanaPD)
install(TARGETS KalthoffWinkler DESTINATION examples)

add_executable(DiskImpact disk_impact.cpp)
target_link_libraries(DiskImpact LINK_PUBLIC CabanaPD)
install(TARGETS DiskImpact DESTINATION examples)
