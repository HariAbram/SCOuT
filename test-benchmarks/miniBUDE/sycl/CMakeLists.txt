cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(bude_sycl)
SET(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SYCL_RUNTIME "HIPSYCL")
set(HIPSYCL_TARGETS generic)

set(HIPSYCL_INSTALL_DIR "/opt/AdaptiveCpp")

if (SYCL_RUNTIME)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

    if (${SYCL_RUNTIME} STREQUAL "HIPSYCL")

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()
        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake/hipSYCL)

        if (NOT EXISTS "${hipSYCL_DIR}")
            message(WARNING "Falling back to hipSYCL < 0.9.0 CMake structure")
            set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake)
        endif ()
        if (NOT EXISTS "${hipSYCL_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for hipSYCL")
        endif ()

        find_package(AdaptiveCpp CONFIG REQUIRED)
     
        set(CMAKE_CXX_STANDARD 17)
        # add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        set(EXTRA_FLAGS -Wno-sign-compare -Wno-stringop-truncation)
    elseif (${SYCL_RUNTIME} STREQUAL "HIPSYCL-NEXT") # legacy, remove when no scripts depend on this 

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()
        set(CMAKE_CXX_STANDARD 17)
        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake/hipSYCL)
        # add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        find_package(hipSYCL CONFIG REQUIRED)
        set(EXTRA_FLAGS -Wno-sign-compare -Wno-stringop-truncation)
    elseif (${SYCL_RUNTIME} STREQUAL "COMPUTECPP")

        if (NOT ComputeCpp_DIR)
            message(FATAL_ERROR "ComputeCpp_DIR is undefined")
        endif ()
        set(OpenCL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
        add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        find_package(ComputeCpp REQUIRED)
        set(COMPUTECPP_USER_FLAGS -O3 -no-serial-memop)
    elseif (${SYCL_RUNTIME} STREQUAL "DPCPP")

        if (NOT DPCPP_BIN)
            message(STATUS "DPCPP_BIN unspecified, using dpcpp on path")
            set(CMAKE_CXX_COMPILER "dpcpp")
        else()
            set(CMAKE_CXX_COMPILER ${DPCPP_BIN})
        endif ()

        if(DPCPP_INCLUDE)
            include_directories(${DPCPP_INCLUDE})
        endif()

        add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
        set(CMAKE_CXX_STANDARD 17)
        set(EXTRA_FLAGS -pedantic)
    else ()
        message(FATAL_ERROR "SYCL_RUNTIME unsupported, must be one of HIPSYCL|HIPSYCL-NEXT|COMPUTECPP|DPCPP, got ${SYCL_RUNTIME}")
    endif ()
else ()
    message(FATAL_ERROR "SYCL_RUNTIME not defined, must be one of HIPSYCL|HIPSYCL-NEXT|COMPUTECPP|DPCPP")
endif ()


set(SOURCES
        src/bude.cpp
        src/vec-pose-inner.cpp)

include_directories(src)
add_executable(bude ${SOURCES})

separate_arguments(CXX_EXTRA_FLAGS)
separate_arguments(CXX_EXTRA_LINKER_FLAGS)

target_compile_options(bude
        PUBLIC
	-O3
        -Wall
        -Wextra
        -Wcast-align
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes

        ${EXTRA_FLAGS}
        )


if (NOT DEFINED NUM_TD_PER_THREAD)
    message(STATUS "NUM_TD_PER_THREAD unspecified, defaulting to 1")
    set(NUM_TD_PER_THREAD 1)
endif ()

if (NOT NUM_TD_PER_THREAD MATCHES "^[0-9]+$")
    message(FATAL_ERROR "NUM_TD_PER_THREAD must be an integer, got ${NUM_TD_PER_THREAD}")
endif ()

add_definitions(-DNUM_TD_PER_THREAD=${NUM_TD_PER_THREAD})

if (USE_PRE_SYCL121R3)
    add_definitions(-DUSE_PRE_SYCL121R3)
endif ()

if (DISABLE_ND_RANGE)
    add_definitions(-DDISABLE_ND_RANGE)
endif ()

set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -Ofast -ffast-math -march=native -mtune=native ${CXX_EXTRA_FLAGS})

target_compile_options(bude PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(bude PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(bude PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")

if (${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message(WARNING "target_link_options is only available in CMake >= 3.13.0, using fallback target_link_libraries, this may cause issues with some compilers")
    message(WARNING "whitespaces are not supported for CXX_EXTRA_LINKER_FLAGS/CXX_EXTRA_FLAGS in this mode as they are treated as libraries arguments (CMake splits them)")
    if (DEFINED CXX_EXTRA_LINKER_FLAGS)
        list(APPEND EXTRA_LINK_FLAGS "-Wl,${CXX_EXTRA_LINKER_FLAGS}")
    endif ()

    target_link_libraries(bude PUBLIC ${EXTRA_LINK_FLAGS})
    target_link_libraries(bude PUBLIC ${CXX_EXTRA_FLAGS})

else ()
    target_link_options(bude PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
    target_link_options(bude PUBLIC ${CXX_EXTRA_FLAGS})
endif ()


if (NOT ${SYCL_RUNTIME} STREQUAL "DPCPP")
    add_sycl_to_target(
            TARGET bude
            SOURCES ${SOURCES}) # must be the last
endif ()

