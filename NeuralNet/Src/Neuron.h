

#ifndef NEURON_H
#define NEURON_H

#define _USE_MATH_DEFINES
#include <stdexcept> 
#include <vector>
#include <cmath>
#include <math.h>
#include <functional>
#include "Layer.h"
#include <fstream>

struct Synapse {
	float f_value_;
	float f_last_diff_;
};

class Layer;

class Neuron
{
private:
	Layer* prev_layer_; //pointer to the previous Layer of Neurons (Null for the Input Layer)
	
	float f_value_;
	float f_error_;
	int index_;
	bool is_bias = false;

	std::vector<Synapse> synaps_vect; // Synapse vector connecting this Neuron with Neurons of previous Layer


public:
//Initializing all properties with nulls
//Creating a Synapse vector of prev_layer_size size
//Initializing all synapses with 0
//Setting index_ property to index value 
	Neuron(int index, int prev_layer_size);

	//Initializes Synapse values for the current Neuron using lambda init_func
	void InitSynaps(std::function<float()> init_func);
	
	void SetPreviousLayerPtr(Layer* layer);

	bool IsBias();
	float GetIndex();
	float GetValue();
	void SetValue(float val);
	float GetError();
	void SetError(float error);

	//Sets the current Neuron Value to 1.0F and is_bias to true
	void MakeBias();

	//Returns the Value of the Synapse connecting the current Neuron with previous layer Neuron with neuron_index
	float GetSynapsValForNeuron(int neuron_index);

	float GetSynapsDiffForNeuron(int neuron_index);

	//Calculates the Value of the current Neuron using it's Synapses and previous Layer Neurons
	//Activates the Value with activation func "act"
	void CalcValue(std::function<float(float)> act);

	//Calculating the difference between the true_value and the calculater value
	//Setting the f_error property using the derivate of activation func (activate_derive_func)
	void CalcOutputError(float f_true_value, std::function <float(float)> atctivate_deriv_func);

	//Adjasting the current value of the Synapse No "index"  by adding "correction"
	//Setting new f_value and f_correction props values
	void CorrectSynapsForNeuron(int index, float correction);
	
	void WriteVectorToFile(std::ofstream& file);
};
#endif