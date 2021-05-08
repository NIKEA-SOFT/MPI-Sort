cmake_minimum_required(VERSION 3.5)

project(MPI_Sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

add_executable(MPI_Sort main.cpp)
target_link_libraries(MPI_Sort ${MPI_LIBRARIES})

set_property(TARGET MPI_Sort PROPERTY CXX_STANDARD 17)
set_property(TARGET MPI_Sort PROPERTY CXX_STANDARD_REQUIRED ON)
