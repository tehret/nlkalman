set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

add_library(nlklib SHARED
	nlkalman.cpp
	LibMatrix.cpp
	iio.c
	LibImages.cpp
	LibVideoT.cpp
	mt19937ar.c
	Utilities.cpp
	parametric_utils.cpp
	parametric_transformation.cpp
	)

set_target_properties(nlklib PROPERTIES
	SOVERSION ${PROJECT_VERSION})

install(TARGETS nlklib
    LIBRARY DESTINATION lib
    COMPONENT libs)
