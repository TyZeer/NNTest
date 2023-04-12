#include "Neuron.h"
float Neuron::GetValue() { return f_value_; }

void Neuron::SetValue(float val) {
	f_value_ = !is_bias ? val : throw std::invalid_argument("You can't set bias neuron value!");
}
float Neuron::GetError() { return f_error_; }

void Neuron::SetError(float error) { f_error_ = error; }

float Neuron::GetSynapsValForNeuron(int neuron_index) { return synaps_vect[neuron_index].f_value_; }

float Neuron::GetSynapsDiffForNeuron(int neuron_index) { return synaps_vect[neuron_index].f_last_diff_; }


void Neuron::MakeBias() {
	f_value_ = 1.0f;
	is_bias = true;
}

void Neuron::InitSynaps(std::function<float()> init_func) 
{
	for (auto& synaps : synaps_vect) {
		synaps.f_value_ = init_func();
	}
}

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

void Neuron::SetPreviousLayerPtr(Layer* layer)
{
	prev_layer_ = layer;
}

/*
Neuron::Neuron(int index, Layer* prev_layer)
{
	f_value_ = 0.0F;
	f_error_ = 0.0F;
	index_ = index;
	for (int i = 0; i < (prev_layer==NULL ? 0 : prev_layer->NeuronsCount()); i++)
	{
		synaps_vect.push_back({ 0.0F, 0.0F });
	}
}*/

void Neuron::CalcValue(std::function<float(float)> act) {
	if (is_bias) return;
	if (prev_layer_ == NULL) return;
	float value_to_activate = 0.0f;
	for (int i = 0; i < prev_layer_->NeuronsCount(); i++)
	{
		value_to_activate += prev_layer_->GetNeuronVal(i)* synaps_vect[i].f_value_;
	}
	f_value_ = act(value_to_activate);
}
void Neuron::CalcOutputError(float true_value, std::function <float(float)> atctivate_deriv_func) {
	f_error_ = (true_value - f_value_) * atctivate_deriv_func(f_value_);
}
//void Neuron::InitSynaps(std::function<float()> func) {
//	for (auto& Synapse : synaps_vect) {
//		Synapse.f_value_ = func();
//	}
//
//}
void Neuron::CorrectSynapsForNeuron(int index, float correction) {
	synaps_vect[index].f_value_ += correction;
	synaps_vect[index].f_last_diff_ = correction;
}
