cmake_minimum_required(VERSION 3.20)
project(three_body_cpu_verification LANGUAGES CXX)

add_executable(three_body_verify verify.cpp)
target_compile_features(three_body_verify PRIVATE cxx_std_20)
target_include_directories(three_body_verify PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
                                                     "${CMAKE_CURRENT_SOURCE_DIR}/../cuda")
set_target_properties(three_body_verify PROPERTIES CXX_EXTENSIONS OFF)

enable_testing()
add_test(NAME projection_parity COMMAND three_body_verify)
