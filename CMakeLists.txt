cmake_minimum_required(VERSION 3.12)

project(NNTest VERSION 1.0.0)

add_library(NeuralNet 
            ./NeuralNet/Src/NeuralNetwork.cpp 
            ./NeuralNet/Src/NeuralNetwork.h 
            ./NeuralNet/Src/Layer.cpp 
            ./NeuralNet/Src/Layer.h 
            ./NeuralNet/Src/Neuron.cpp 
            ./NeuralNet/Src/Neuron.h 
            ./NeuralNet/Src/INeuralNetworkParametres.cpp 
            ./NeuralNet/Src/INeuralNetworkParametres.h
            )

add_executable(NeuralTest 
            ./NetworkTest/Src/NNData.cpp 
            ./NetworkTest/Src/NNData.h 
            ./NetworkTest/Src/NeuralNetworkParametres.cpp 
            ./NetworkTest/Src/NeuralNetworkParametres.h 
            ./NetworkTest/Src/main.cpp  )

target_link_libraries(NeuralTest PRIVATE NeuralNet)
