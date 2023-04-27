

#include "NeuralNetworkParametres.h"

float NNParametres::SetNeuralNetworkMomentum() { return 0.3F; }
float NNParametres::SetNeuralNetworkLearnRate() { return 0.7F; }
bool NNParametres::DoAddBias() { return false; }
void NNParametres::SetNeuralNetworkConfiguration(std::vector<int>& NetConfig) {
	NetConfig.push_back(2);
	NetConfig.push_back(2);
	NetConfig.push_back(1);
	
	//NetConfig.push_back(10);
}
