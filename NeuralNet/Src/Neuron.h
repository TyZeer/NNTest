

#ifndef NEURON_H
#define NEURON_H

#define _USE_MATH_DEFINES
#include <stdexcept> 
#include <vector>
#include <cmath>
#include <math.h>
#include <functional>
#include "Layer.h"


struct Synapse {
	float f_value_;
	float f_last_diff_;
};

class Layer;

class Neuron
{
private:
	Layer* prev_layer_;
	
	float f_value_;
	float f_error_;


public:
	std::vector<Synapse> synaps_vect; // PUT BACK TO PRIVATE

	bool is_bias = false;
	int index_;
	void InitSynaps(std::function<float()> init_func);


	//Neuron(int index, Layer * prev_layer = NULL);
	Neuron(int index, int prev_layer_size);
	//void SetActivationFunc(std::function<float(float)> act) { activate_func = act; }
	//void SetActDerivFunc(std::function<float(float)> act_deriv) { atctivate_deriv_func = act_deriv; }
	void SetPreviousLayerPtr(Layer* layer);

	float GetValue();

	void SetValue(float val);

	float GetError();

	void SetError(float error);

	//Neuron(int synaps_size, int index, vector <Sinapse> * sin_vect = NULL);
	void MakeBias();

	float GetSynapsValForNeuron(int neuron_index);

	float GetSynapsDiffForNeuron(int neuron_index);


	void CalcValue(std::function<float(float)> act);

	void CalcOutputError(float f_true_value, std::function <float(float)> atctivate_deriv_func);

	void CorrectSynapsForNeuron(int index, float correction);

	float ReturnSynapseVal(int index) { return  synaps_vect[index].f_value_; }
	
	Synapse* ReturnVectPointer() { return &synaps_vect[0]; }
	//хз, правильно ли это, далльше видимо надо будет писать сам сейв и лоад
};
#endif