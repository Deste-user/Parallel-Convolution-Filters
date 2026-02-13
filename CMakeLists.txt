cmake_minimum_required(VERSION 3.10)
project(ConvolutionProject LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


add_executable(main_cuda main.cu)

set_property(TARGET main_cuda PROPERTY CUDA_ARCHITECTURES 100)
target_include_directories(main_cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(main_cuda PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main_cuda ${OpenCV_LIBS})


set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(benchmark ${OpenCV_LIBS})