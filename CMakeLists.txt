cmake_minimum_required(VERSION 3.16)
project(object_detection_lib_all)
include(CheckLanguage)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message("-- OK: Found CUDA Compiler, CUDA is enabled")
    set(ENABLE_CUDA ON)
    set(ENABLE_CUDNN ON)
else()
    message("-- Warning: CUDA Compiler not found, CUDA is disabled ( training will be slow )")
    set(ENABLE_CUDA OFF)
    set(ENABLE_CUDNN OFF)
endif()

find_package(OpenCV)
if(OpenCV_FOUND)
    message("-- OK: Found OpenCV, OpenCV is enabled")
    set(ENABLE_OPENCV ON)
else()
    message("-- Warning: OpenCV not found, OpenCV is disabled")
    set(ENABLE_OPENCV OFF)
endif()

add_subdirectory(3rdparty/darknet)
add_subdirectory(src)

set_target_properties(dark PROPERTIES COMPILE_FLAGS "-w") # darknet ignores warnings..... so disable the spam
