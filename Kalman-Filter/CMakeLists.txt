cmake_minimum_required(VERSION 3.5)
project(OpenCVKF)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
add_executable(OpenCVKF main.cpp)
target_link_libraries( OpenCVKF ${OpenCV_LIBS} )