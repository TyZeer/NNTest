#pragma once
#include <vector>
class INeuralNetworkParametres
{
public:
	virtual float SetNeuralNetworkMomentum() = 0;
	virtual float SetNeuralNetworkLearnRate() = 0;
	virtual bool DoAddBias() { return false; }
	virtual void SetNeuralNetworkConfiguration(std::vector<int>& NetConfig) = 0;

};

