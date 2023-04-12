#include "Neuron.h"
#include "Layer.h"

//Neuron* GetNeuron(int neuron_index) { return &neurons_vect_[neuron_index]; } //что здесь писать

rsize_t Layer::NeuronsCount() { return neurons_vect_.size(); } // тут тоже не понял

float Layer::GetNeuronVal(int neuron_index) { return neurons_vect_[neuron_index].GetValue(); }

void Layer::SetNeuronError(int position, float value_to_set) { neurons_vect_[position].SetError(value_to_set); }

void Layer::SetNeuronValue(int position, float value_to_set) { neurons_vect_[position].SetValue(value_to_set); }

void Layer::CalculateNeuron(int position, std::function<float(float)> act)
{
	neurons_vect_[position].CalcValue(act);
}


Layer::Layer(Layerposition pos, int layer_size, bool do_add_bias, int prev_layer_size) {
	position_ = pos;
	layersize_ = layer_size;
	prev_layer_ = NULL;
	for (int i = 0; i < layersize_; i++)
	{
		neurons_vect_.push_back( Neuron(i, prev_layer_size ));
	}
	if (do_add_bias) {
		neurons_vect_.push_back(Neuron(layersize_,0));
		neurons_vect_[layersize_].MakeBias();
	}

}

void Layer::SetPreviousLayerPtr(Layer* prev)
{
	prev_layer_ = prev;
	for (size_t i = 0; i < layersize_; i++)
	{
		neurons_vect_[i].SetPreviousLayerPtr(prev);
	}
}

void Layer::InitSynapse(std::function<float()> func) {
	for (size_t i = 0; i < neurons_vect_.size(); i++)
	{
		neurons_vect_[i].InitSynaps(func);
	}
}
void Layer::BackPropogate(std::function<float(float)> act_deriv, float Learn_rate, float Momentum) 
{
	

	for (auto& prev_layer_neuron : prev_layer_->neurons_vect_)  // Для каждого нейрона предыдущего слоя
	{
		float f_error = 0.0F;
		float f_gradient = 0.0F;
		float f_correction = 0.0F;
		for (auto& this_layer_neuron : neurons_vect_) // Для каждого нейолна этого слоя
		{
			if (this_layer_neuron.is_bias) {
				continue;
			}//разобраться с синапсами нейрона bias

			f_error += this_layer_neuron.GetError() * this_layer_neuron.GetSynapsValForNeuron(prev_layer_neuron.index_);
		}

		prev_layer_neuron.SetError(f_error * act_deriv(prev_layer_neuron.GetValue()));
		
		// Коррекция синапсов, между текущим слоем и нейроном предыдущего слоя
		for (auto& this_layer_neuron : neurons_vect_)
		{
			if (this_layer_neuron.is_bias) continue;

			f_gradient = this_layer_neuron.GetError() * prev_layer_neuron.GetValue();
			f_correction = Learn_rate * f_gradient + Momentum * this_layer_neuron.GetSynapsDiffForNeuron(prev_layer_neuron.index_);

			this_layer_neuron.CorrectSynapsForNeuron(prev_layer_neuron.index_, f_correction);
		}
	}
}
