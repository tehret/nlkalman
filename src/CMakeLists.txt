set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/nlkalman)

add_executable(nlkalman main_nlkalman.cpp)

include_directories(${LAPACK_INCLUDES})
include_directories(${CBLAS_INCLUDE_DIR})

link_directories(${CMAKE_BINARY_DIR}/lib)
target_link_libraries(nlkalman nlklib tiff jpeg png ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES})
