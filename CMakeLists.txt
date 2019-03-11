
project(NLKalman)

message(STATUS "Configuring main project: " ${CMAKE_PROJECT_NAME})

set(PROJECT_VERSION_MAJOR  2)
set(PROJECT_VERSION_MINOR  0)
set(PROJECT_VERSION_PATCH  0)

set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    CACHE INTERNAL "The project version.")

###############################################################################

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ggdb")

set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lfftw3f -fsanitize=address -static-libasan")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lfftw3f -O6 -w -s -ffast-math -fno-finite-math-only -fomit-frame-pointer -fstrength-reduce")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

###############################################################################

cmake_minimum_required(VERSION 2.8.2)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")
#set(CMAKE_VERBOSE_MAKEFILE ON)

###############################################################################

find_package( CBLAS REQUIRED )
find_package( LAPACKE REQUIRED )
find_package (TIFF REQUIRED)
find_package (JPEG REQUIRED)
find_package (PNG REQUIRED)
include_directories (PUBLIC ${TIFF_INCLUDE_DIR} PUBLIC ${JPEG_INCLUDE_DIR} PUBLIC ${PNG_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
#	message("-- OpenMP found.")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###############################################################################

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/nlkalman)
add_subdirectory(src)
add_subdirectory(DCTdenoising)
add_subdirectory(tvl1flow)
