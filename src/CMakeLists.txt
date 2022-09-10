cmake_minimum_required(VERSION 3.16)
project(object_detection_lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package(CURL REQUIRED)
find_package(TBB)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python3_FiftyOne)
find_package(Minizip) # sudo apt install libminizip-dev

if(Python3_Development_FOUND)
    if(Python3_FiftyOne_FOUND)
        add_definitions(-DPYTHON3_FOUND)
        message("-- OK: Found python-fiftyone, fiftyone is enabled ( downloading google-open-images training sets is enabled )")
    else()
        message("-- Warning: python-fiftyone not found, fiftyone is disabled ( downloading google-open-images training sets is disabled. you can install it using 'python -m pip install fiftyone' )")
    endif()
else()
    message("-- Warning: Python dev not found, Python is disabled ( downloading google-open-images training sets is disabled. you can enable it using 'sudo apt install libpython3-dev' )")
endif()

if(MINIZIP_FOUND)
    add_definitions(-DMINIZIP_FOUND)
    message("-- OK: Found Minizip, Zipping files is enabled ( sharing images/annotation to google colab is enabled )")
else()
    message("-- Warning: Minizip not found, Zipping files is disabled ( sharing images/annotation to google colab is disabled. you can enable it using 'sudo apt install libminizip-dev' )")
endif()

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -pedantic -Werror)

file(GLOB_RECURSE sources_lib src_lib/*.c src_lib/*.cpp src_lib/*.hpp)
file(GLOB_RECURSE sources_cli src_cli/*.c src_cli/*.cpp src_cli/*.hpp)

include_directories(../include)
include_directories(../3rdparty/stb/include)
include_directories(../3rdparty/darknet/include)
include_directories(../3rdparty/httplib)
include_directories(${CURL_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${MINIZIP_INCLUDE_DIRS})

add_library(object_detection_lib STATIC ${sources_lib})
add_executable(object_detection_cli ${sources_cli})

target_link_libraries(object_detection_lib PRIVATE dark ${CURL_LIBRARIES} ${TBB_LIBRARIES} ${Python3_LIBRARIES} ${MINIZIP_LIBRARIES})
target_link_libraries(object_detection_cli PRIVATE stdc++ m object_detection_lib)
