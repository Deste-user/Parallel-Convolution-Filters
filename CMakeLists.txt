cmake_minimum_required(VERSION 3.18)
project(ConvolutionProject LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(main_cuda main.cu)


set_property(TARGET main_cuda PROPERTY CUDA_ARCHITECTURES 120)
target_include_directories(main_cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(main_cuda PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main_cuda ${OpenCV_LIBS})



add_executable(benchmark benchmark.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(benchmark PRIVATE -O3 -march=native)
endif()

target_include_directories(benchmark PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(benchmark ${OpenCV_LIBS})