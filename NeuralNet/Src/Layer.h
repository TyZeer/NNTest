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
	std::vector<Neuron> neurons_vect_; 
	Layer* prev_layer_;

public:
	Layer(Layerposition, int layer_size, bool do_add_bias, int prev_layer_size);

	//Returns the size of Neurons vector (including bias)
	size_t NeuronsCount(); 

	//Returns the value of the current Layer Neuron with index "neuron_index"
	float GetNeuronVal(int neuron_index); 
	
	//Transmitting the previous layer pointer to every Neuron in this Layer
	void SetPreviousLayerPtr(Layer* prev);
	
	//Calling InitSynapse using lambda for every Neuron within this Layer
	void InitSynapse(std::function<float()>);

	//Calling Calculate method for the current Layer Neuron in position "position"
	void CalculateNeuron(int position, std::function<float(float)> act);

	//Error Back Propagation algorithm for the Layers Synapses 
	void BackPropogate(std::function<float(float)> act_deriv, float Learn_rate, float Momentum);

	//Setters for Neuron Value and Error
	void SetNeuronValue(int position, float value_to_set);
	void SetNeuronError(int position, float value_to_set);

};

#endif