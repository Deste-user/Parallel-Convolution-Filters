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

add_executable(main_seq_aos Layout_AoS_Convolution.cpp)
target_link_libraries(main_seq_aos ${OpenCV_LIBS})

add_executable(main_seq_soa Layout_SoA_Convolution.cpp)
target_include_directories(main_seq_soa PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main_seq_soa ${OpenCV_LIBS})