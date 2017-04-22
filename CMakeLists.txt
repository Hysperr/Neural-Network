cmake_minimum_required(VERSION 3.7)
project(Neural_Network)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        Green.cpp
        Green.h
        main.cpp
        Node.cpp
        Node.h
        optdigits_train.txt
        optdigits_names.txt
        optdigits_test.txt
        cmake-build-debug/tester.cpp)

add_executable(Neural_Network ${SOURCE_FILES})