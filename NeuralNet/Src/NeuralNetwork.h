#pragma once
#include "INeuralNetworkParametres.h"
#include <vector>
#include "Neuron.h"
#include "Layer.h"
#include <iostream>

class NeuralNetwork
{
private:
	float Learn_rate;
	float Momentum;
	bool Do_add_bias;
	bool Isinitialized = false;
	std::vector<Layer> Layer_vect_;
	std::vector<int> Network_configuration;
	std::vector<float> OutputDifference;
public:
	NeuralNetwork();
	NeuralNetwork(INeuralNetworkParametres& params);
	void InitSynapseInLayer(int layernum, std::function<float()> Initfunc);
	void SetSynapseByIndex(int layernum, int neuronnum, int synapse_index, float value); //Для тестирования
	void SetInputNeuron(int position, float value);
	void Calculate(std::function<float(float)> act);
	void CalcOutputError(int position, float true_value, std::function<float(float)> act_deriv);
	void BackPropogationError(std::function<float(float)> act_deriv);
	float MediumSquareError();
	void ShowNeuralNetworkParametres(std::ostream& output);
	float GetOutputNeuronValue(int neuron_index);
	void SetMomentum(float moment);
	void SetLearnRate(float rate) { Learn_rate = rate; }
	void AddBias(bool bias) { Do_add_bias = bias; }
	void SetLayer(std::vector<int>& Configuration);
	//void Save(std::string path);
	//void Load(std::string path);
};


 