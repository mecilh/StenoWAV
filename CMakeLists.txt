cmake_minimum_required(VERSION 3.14)
project(StenoWAV VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stenowav
    src/main.cpp
    src/wav.cpp
    src/steno.cpp
)

target_include_directories(stenowav PRIVATE include)
