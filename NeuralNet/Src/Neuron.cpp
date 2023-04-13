#include "Neuron.h"

//Initializing all properties with nulls
//Creating a Synapse vector of prev_layer_size size
//Initializing all synapses with 0
//Setting index_ property to index value
Neuron::Neuron(int index, int prev_layer_size)
{
	f_value_ = 0.0F;
	f_error_ = 0.0F;
	index_ = index;
	prev_layer_ = NULL;
	for (int i = 0; i < prev_layer_size; i++)
	{
		synaps_vect.push_back({ 0.0F, 0.0F });
	}

}

bool Neuron::IsBias() {return is_bias;}

// Returns the current Neuron index
float Neuron::GetIndex(){ return index_;}

//Returns the Value for tht curent Neuron
float Neuron::GetValue() { return f_value_; }

//Sets the Value for the current Neuron if it's not a Bias one
void Neuron::SetValue(float val) {
	f_value_ = !is_bias ? val : throw std::invalid_argument("You can't set bias neuron value!");
}

//Returns the Error of the current Neuron
float Neuron::GetError() { return f_error_; }

//Sets the Error of the current Neuron
void Neuron::SetError(float error) { f_error_ = error; }

//Returns the Value of the Synapse connecting the current Neuron with prevoous layer Neuron with neuron_index
float Neuron::GetSynapsValForNeuron(int neuron_index) { return synaps_vect[neuron_index].f_value_; }

//Returns the Diggerence calculated for the Synapse connecting the current Neuron with prevoous layer Neuron with neuron_index
float Neuron::GetSynapsDiffForNeuron(int neuron_index) { return synaps_vect[neuron_index].f_last_diff_; }

//Sets the current Neuron Value to 1.0F and is_bias to true
void Neuron::MakeBias() {
	f_value_ = 1.0f;
	is_bias = true;
}

//Initializes Synapse values for the current Neuron using lambda init_func
void Neuron::InitSynaps(std::function<float()> init_func) 
{
	for (auto& synaps : synaps_vect) {
		synaps.f_value_ = init_func();
	}
}


//Sets the pointer to the previous Layer after all layers was appended to the Layers Vector (in NeuralNetwork obj)
void Neuron::SetPreviousLayerPtr(Layer* layer)
{
	prev_layer_ = layer;
}

//Calculates the Value of the current Neuron using it's Synapses and previous Layer Neurons
//Activates the Value with activation func "act"
void Neuron::CalcValue(std::function<float(float)> act) {

	if (is_bias) return; //Don't need to calculate the bias neuron value
	if (prev_layer_ == NULL) return; //Means that current Neuron belongs to the Input Layer and can't be calculated

	float value_to_activate = 0.0f;
	// Calculating the Value
	for (int i = 0; i < prev_layer_->NeuronsCount(); i++)
	{
		value_to_activate += prev_layer_->GetNeuronVal(i)* synaps_vect[i].f_value_;
	}
	//Activating the calculated value and setting the f_value property
	f_value_ = act(value_to_activate);
}

//Calculating the difference between the true_value and the calculater value
//Setting the f_error property using the derivate of activation func (activate_derive_func)
void Neuron::CalcOutputError(float true_value, std::function <float(float)> atctivate_deriv_func) {
	f_error_ = (true_value - f_value_) * atctivate_deriv_func(f_value_);
}

//Adjasting the current value of the Synapse No "index"  by adding "correction"
//Setting new f_value and f_correction props values
void Neuron::CorrectSynapsForNeuron(int index, float correction) {
	synaps_vect[index].f_value_ += correction;
	synaps_vect[index].f_last_diff_ = correction;
}
