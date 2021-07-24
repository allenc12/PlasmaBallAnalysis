cmake_minimum_required(VERSION 3.10)

project(cap_video)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

find_package(jsoncpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${jsoncpp_INCLUDE_DIRS})

add_executable(cap_video main.cpp VideoCap.cpp)

target_link_libraries(cap_video ${OpenCV_LIBS} ${jsoncpp_LIBS})
