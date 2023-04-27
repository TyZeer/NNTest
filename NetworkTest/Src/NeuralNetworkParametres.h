#pragma once
#include "NeuralNetwork.h"
#include <vector>
class NNParametres : public INeuralNetworkParametres
{
public:
	
	virtual float SetNeuralNetworkMomentum();
	virtual float SetNeuralNetworkLearnRate();
	virtual bool DoAddBias();
	virtual void SetNeuralNetworkConfiguration(std::vector<int>& NetConfig);
	
};


