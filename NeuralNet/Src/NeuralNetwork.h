#pragma once
#include <vector>
#include <iostream>
#include "memory"
#include <functional>

class INeuralNetworkParametres
{
public:
	virtual float SetNeuralNetworkMomentum() = 0;
	virtual float SetNeuralNetworkLearnRate() = 0;
	virtual bool DoAddBias() { return false; }
	virtual void SetNeuralNetworkConfiguration(std::vector<int>& NetConfig) = 0;

};


class NeuralNetwork
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;
public:
	// 1st scenario:
	// Creating an empty object
	NeuralNetwork();
	~NeuralNetwork();
	// Setting it's parameters
	NeuralNetwork& SetMomentum(float momentum); //{ Momentum = momentum; return *this; }
	NeuralNetwork& SetLearnRate(float rate); //{ Learn_rate = rate; return *this; }
	NeuralNetwork& AddBias(bool bias);
	NeuralNetwork& SetLayers(const std::vector<int>& Configuration);
	
	// 2nd scenario
	// Creating a full Neural Network using INeuralNetworkParametres object
	NeuralNetwork(INeuralNetworkParametres& params);

	// Initializing synapses in Layer "layernum" with lambda Initfunc
	void InitSynapseInLayer(int layernum, std::function<float()> Initfunc);
	//void SetSynapseByIndex(int layernum, int neuronnum, int synapse_index, float value); //��� ������������
	
	// Setting value for the Neuron in position "position" with "value" for Imput Layer
	void SetInputNeuron(int position, float value);

	// Calculate the Neural Network
	void Calculate(std::function<float(float)> act);

	void CalcOutputError(int position, float true_value, std::function<float(float)> act_deriv);
	void BackPropogationError(std::function<float(float)> act_deriv);
	float MeanSquaredError();
	void ShowNeuralNetworkParametres(std::ostream& output);
	float GetOutputNeuronValue(int neuron_index);

	//Setters
	//void SetMomentum(float momentum) {Momentum = momentum;}
	//void SetLearnRate(float rate) { Learn_rate = rate; }
	//void AddBias(bool bias);
	
	
	//void Save(std::string path);
	//void Load(std::string path);
};


 