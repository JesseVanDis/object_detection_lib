cmake_minimum_required(VERSION 3.21)
project(yolo)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package(CURL REQUIRED)
find_package(TBB)

if(TBB_FOUND)
    message("-- OK: Found TBB, TBB is enabled")
    set(TBB_LIBRARIES TBB::tbb)
else()
    message("-- Warning: TBB not found, TBB is disabled ( initial setup will be slower )")
endif()

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -pedantic -Werror)

file(GLOB_RECURSE sources *.cpp *.hpp)

include_directories(../3rdparty/stb/include)
include_directories(${CURL_INCLUDE_DIR})

add_library(yolo ${sources})
target_link_libraries(yolo PRIVATE dark ${CURL_LIBRARIES} ${TBB_LIBRARIES})