project(FluidSim)
cmake_minimum_required(VERSION 3.1)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

option(GUI "Runs the program with an OpenGL GUI" OFF)
if(GUI)
	find_package(OpenGL REQUIRED)

	set(GLFW_INCLUDE_PATH "" CACHE PATH "The directory that contains GL/glfw.h" )
	set(GLFW_LIBRARY "" CACHE FILEPATH "libglfw.a or glfw.lib" )

	set(GLEW_INCLUDE_PATH "" CACHE PATH "The directory that contains GL/glew.h" )
	set(GLEW_LIBRARY "" CACHE FILEPATH "libglew.a or glew.lib" )
	
	add_definitions(-DWITH_GUI)
endif()

#------------------------------------------------------------------------------
# OVERWRITE GLOBAL FindMATOG.cmake!
#------------------------------------------------------------------------------
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)
find_package(MATOG REQUIRED)

file (GLOB_RECURSE includes *.h)
file (GLOB sources *.cc)

# Specify the files that need to be compiled and linked.
# This will create an executable named 'fluidsim'.
cuda_add_executable(fluidsim ${sources} ${includes})

# # Fermi
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_21")

# Kepler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

# # Maxwell
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

# MATOG Genertated Code
ADD_SUBDIRECTORY ( ${CMAKE_SOURCE_DIR}/matog_gen )

SET ( KERNEL_FILES
    fluidSimKernel.cu
)
SET_SOURCE_FILES_PROPERTIES (
	${KERNEL_FILES}
	PROPERTIES HEADER_FILE_ONLY False
)

ADD_CUSTOM_TARGET ( copy DEPENDS ${KERNEL_FILES})
FOREACH ( KERNEL_FILE ${KERNEL_FILES})
	ADD_CUSTOM_COMMAND(TARGET copy COMMAND ${CMAKE_COMMAND}
	-E copy_if_different "${CMAKE_SOURCE_DIR}/${KERNEL_FILE}" "${CMAKE_BINARY_DIR}/${KERNEL_FILE}")
ENDFOREACH ()
ADD_CUSTOM_COMMAND(TARGET copy COMMAND ${CMAKE_COMMAND} 
    -E copy "${CMAKE_SOURCE_DIR}/matog.config.json" "${CMAKE_BINARY_DIR}/matog.json")

FILE (GLOB LIB_FILES "matog_gen/*.cu")
FILE (COPY ${LIB_FILES} DESTINATION "${CMAKE_BINARY_DIR}")
FILE (COPY "matog_gen/matog.db" DESTINATION "${CMAKE_BINARY_DIR}")

# Activate (and require) C++11 support
set_property(TARGET fluidsim PROPERTY CXX_STANDARD 11)
set_property(TARGET fluidsim PROPERTY CXX_STANDARD_REQUIRED ON)
# Activate (and require) C++11 support
set_property(TARGET matog_gen PROPERTY CXX_STANDARD 11)
set_property(TARGET matog_gen PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries (fluidsim
    ${CUDA_CUDA_LIBRARY}
	${MATOG_LIBRARIES}
	matog_gen
)
SET(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include" ${CUDA_TOOLKIT_INCLUDE})

include_directories (
	${CUDA_INCLUDE_DIR}
	${MATOG_INCLUDE_DIR}
)

if(GUI)
	target_link_libraries (fluidsim
		${OPENGL_LIBRARY}
		${GLFW_LIBRARY}
		${GLEW_LIBRARY}
	)
    include_directories (
		${GLFW_INCLUDE_PATH}
		${GLEW_INCLUDE_PATH}
	)
endif()

# Instruct CMake to set the correct paths in the bsub job script and to copy it
# to the directory that will contain the executable.
configure_file(job.sh.in job.sh @ONLY)
configure_file(cudaProf.sh.in cudaProf.sh @ONLY)
