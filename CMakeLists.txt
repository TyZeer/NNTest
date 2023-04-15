cmake_minimum_required(VERSION 3.12)

project(NNTest VERSION 1.0.0)

option(USE_STATIC "Use static library" ON )
message(STATUS "OS: ${CMAKE_SYSTEM_NAME}")

add_library(NeuralNet 
            ./NeuralNet/Src/NeuralNetwork.cpp 
            ./NeuralNet/Src/NeuralNetwork.h 
            ./NeuralNet/Src/Layer.cpp 
            ./NeuralNet/Src/Layer.h 
            ./NeuralNet/Src/Neuron.cpp 
            ./NeuralNet/Src/Neuron.h 
            )

add_executable(NeuralTest 
            ./NetworkTest/Src/NNData.cpp 
            ./NetworkTest/Src/NNData.h 
            ./NetworkTest/Src/NeuralNetworkParametres.cpp 
            ./NetworkTest/Src/NeuralNetworkParametres.h 
            ./NetworkTest/Src/main.cpp  
            ./NetworkTest/Src/NeuralNetwork.h
            )

target_link_libraries(NeuralTest PRIVATE NeuralNet)
