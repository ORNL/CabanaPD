add_executable(ElasticWave elastic_wave.cpp)
target_link_libraries(ElasticWave LINK_PUBLIC CabanaPD)
install(TARGETS ElasticWave DESTINATION examples)
