cmake_minimum_required(VERSION 3.22)
project(ex3_new)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(ex3_new MapReduceClient.h MapReduceFramework.h SampleClient.cpp MapReduceClient.cpp
        MapReduceFramework.cpp Barrier.cpp Barrier.h)
