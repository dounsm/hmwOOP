cmake_minimum_required(VERSION 4.0)

project(MyProject LANGUAGE CXX VERSION 1.0)

add_subdirectory(modules/replxx EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 11)