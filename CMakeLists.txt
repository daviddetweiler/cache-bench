cmake_minimum_required(VERSION 3.10)

project(cachebench)

find_package(Threads REQUIRED)

add_executable(bench main.cpp)
target_link_libraries(bench Threads::Threads)
