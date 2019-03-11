include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/nlkalman)

add_executable(nlkalman main_nlkalman.cpp)

include_directories(${LAPACK_INCLUDES})
include_directories(${CBLAS_INCLUDE_DIR})

link_directories(${CMAKE_BINARY_DIR}/lib)
target_link_libraries(nlkalman nlklib tiff jpeg png ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES})

configure_file(script.sh ${CMAKE_BINARY_DIR}/bin/script.sh COPYONLY)
configure_file(dct.sh ${CMAKE_BINARY_DIR}/bin/dct.sh COPYONLY)
