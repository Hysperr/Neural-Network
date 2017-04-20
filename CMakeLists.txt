cmake_minimum_required(VERSION 3.7)
project(Neural_Network)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp Node.cpp Node.h Green.cpp Green.h)
add_executable(Neural_Network ${SOURCE_FILES})