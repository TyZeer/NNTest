#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"
enum class Layerposition { Input, Hidden, Output };

class Neuron;

class Layer
{
private:
	int layersize_;
	Layerposition position_;
	
	Layer* prev_layer_;

public:
	std::vector<Neuron> neurons_vect_; // Don't forget to put back to private!!!
	rsize_t NeuronsCount();
	//Neuron* GetNeuron();
	float GetNeuronVal(int neuron_index);
	int GetTrueSize() { return neurons_vect_.size(); }
	Layer(Layerposition, int layer_size, bool do_add_bias, int prev_layer_size);
	void SetPreviousLayerPtr(Layer* prev);
	//void SetActivationFunc(std::function<float(float)>);
	//void SetActDerivFunc(std::function<float(float)>);
	void InitSynapse(std::function<float()>);
	void BackPropogate(std::function<float(float)> act_deriv, float Learn_rate, float Momentum);
	void SetNeuronValue(int position, float value_to_set);
	void SetNeuronError(int position, float value_to_set);
	void CalculateNeuron(int position, std::function<float(float)> act);
};

#endif