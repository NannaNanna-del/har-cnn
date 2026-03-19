cmake_minimum_required(VERSION 3.25)
project(har-cnn
    VERSION 0.1.0
    DESCRIPTION "Human Action Recognition using CNN"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror=return-type
)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Source files
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Library
add_library(har_cnn_lib STATIC ${SOURCES})
target_include_directories(har_cnn_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Main executable
add_executable(har_cnn main.cpp)
target_link_libraries(har_cnn PRIVATE har_cnn_lib)

# Tests 
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS har_cnn DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
