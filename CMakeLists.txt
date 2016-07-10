cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
file(GLOB srcs *.cpp *.h*)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
endif (CMAKE_COMPILER_IS_GNUCXX)

add_executable(simple-meanshift-traker ${srcs})
target_link_libraries(simple-meanshift-traker ${OpenCV_LIBS})