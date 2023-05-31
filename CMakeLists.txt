#    cmake_minimum_required(VERSION 3.22)
#
#
#    # Project configuration
#project(ex3 VERSION 1.0 LANGUAGES C CXX)
#
#
## Ensure system has pthreads
#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)
#
#add_library(MapReduceFramework
#        MapReduceClient.h
#        MapReduceFramework.cpp MapReduceFramework.h Barrier.h Barrier.cpp
#        # ------------- Add your own .h/.cpp files here -------------------
#        )
#
#
#set_property(TARGET MapReduceFramework PROPERTY CXX_STANDARD 11)
#target_compile_options(MapReduceFramework PUBLIC -Wall -Wextra)
#target_include_directories(MapReduceFramework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
## link pthreads to your framework
#target_link_libraries(MapReduceFramework PUBLIC Threads::Threads)
#
## Add tests
#add_subdirectory(mattanTests)

#-----------------------------------------------------------------------------------------------

#cmake_minimum_required(VERSION 3.22)
#project(ex3_new)
#
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#
#add_executable(ex3_new
##        SampleClient.cpp
#        MapReduceClient.h
#        MapReduceFramework.h
#        SampleClient.cpp
#        MapReduceClient.cpp
#        MapReduceFramework.cpp
#        Barrier.cpp
#        Barrier.h)

#-----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.22)
project(ex3_new)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(ex3_new
        MapReduceClient.h
        MapReduceFramework.h
        MapReduceClient.cpp
        MapReduceFramework.cpp
        Barrier.cpp
        Barrier.h
        SampleClient.cpp
        )