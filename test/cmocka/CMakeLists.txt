project(obs-cmocka)

find_package(CMocka CONFIG REQUIRED)

# Newer dependency bundles export an imported target instead of variables.
if(TARGET cmocka::cmocka)
  set(CMOCKA_LIBRARIES cmocka::cmocka)
endif()

# Serializer test
add_executable(test_serializer test_serializer.c)
target_include_directories(test_serializer PRIVATE ${CMOCKA_INCLUDE_DIR})
target_link_libraries(test_serializer PRIVATE OBS::libobs ${CMOCKA_LIBRARIES})
add_test(NAME test_serializer COMMAND $<TARGET_FILE:test_serializer>)

# darray test
add_executable(test_darray test_darray.c)
target_include_directories(test_darray PRIVATE ${CMOCKA_INCLUDE_DIR})
target_link_libraries(test_darray PRIVATE OBS::libobs ${CMOCKA_LIBRARIES})

add_test(NAME test_darray COMMAND $<TARGET_FILE:test_darray>)

# bitstream test
add_executable(test_bitstream test_bitstream.c)
target_include_directories(test_bitstream PRIVATE ${CMOCKA_INCLUDE_DIR})
target_link_libraries(test_bitstream PRIVATE OBS::libobs ${CMOCKA_LIBRARIES})

add_test(NAME test_bitstream COMMAND $<TARGET_FILE:test_bitstream>)

# OS path test
add_executable(test_os_path test_os_path.c)
target_include_directories(test_os_path PRIVATE ${CMOCKA_INCLUDE_DIR})
target_link_libraries(test_os_path PRIVATE OBS::libobs ${CMOCKA_LIBRARIES})

add_test(NAME test_os_path COMMAND $<TARGET_FILE:test_os_path>)

# Include the production implementation to exercise its private DSP state directly.
add_executable(test_limiter_v2 test_limiter_v2.c)
target_include_directories(test_limiter_v2 PRIVATE ${CMOCKA_INCLUDE_DIR})
target_link_libraries(test_limiter_v2 PRIVATE OBS::libobs ${CMOCKA_LIBRARIES})
if(UNIX)
  target_link_libraries(test_limiter_v2 PRIVATE m)
endif()
add_test(NAME test_limiter_v2 COMMAND $<TARGET_FILE:test_limiter_v2>)
