cmake_minimum_required(VERSION 2.8)
project( superstitchous )

#Dirty Hacks
set(TMINRES "/home/daniel/Desktop/superstitchous2.0/vsproject/superstitchous2013")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
#
find_package( PkgConfig )
pkg_check_modules(EIGEN3 REQUIRED eigen3)
FIND_PACKAGE(FFTW REQUIRED)
FIND_PACKAGE(TIFF REQUIRED)
FIND_PACKAGE(OpenCV 3 REQUIRED)
#pkg_check_modules(OPENCV REQUIRED opencv)
find_package(Boost COMPONENTS system filesystem REQUIRED)
if (false) #for debugging
	get_cmake_property(_variableNames VARIABLES)
	foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
	endforeach()
endif()
MESSAGE(STATUS "TIMINRES:" ${TMINRES})
INCLUDE_DIRECTORIES(${TMINRES})
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${EIGEN3_LIBRARY_DIRS})


message(STATUS "OPENCV_DIR: " ${OPENCV_INCLUDE_DIRS} )
#message(STATUS "Opencv_LIBS: " ${OpenCV_LIBS} )
set(SOURCES main.cpp common.cpp fetcher.cpp lsqr_int.cpp optimize.cpp
	stitch.cpp tiffint.cpp translate.cpp)
add_executable( stiching ${SOURCES} )
target_link_libraries( stiching  ${OpenCV_LIBS} )
target_link_libraries( stiching  ${Boost_LIBRARIES} )
target_link_libraries( stiching ${TIFF_LIBRARIES} )
target_link_libraries( stiching ${FFTW_LIBRARIES} )
target_link_libraries( stiching "libfftw3f_threads.so")
SET(CMAKE_CXX_FLAGS "-std=c++11 -g -mtune=native -march=native -O3 -Wfatal-errors")
