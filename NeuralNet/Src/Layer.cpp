#include "Neuron.h"
#include "Layer.h"


size_t Layer::NeuronsCount() { return neurons_vect_.size(); }

float Layer::GetNeuronVal(int neuron_index) { return neurons_vect_[neuron_index].GetValue(); }

void Layer::SetNeuronError(int position, float value_to_set) { neurons_vect_[position].SetError(value_to_set); }

void Layer::SetNeuronValue(int position, float value_to_set) { neurons_vect_[position].SetValue(value_to_set); }

void Layer::CalculateNeuron(int position, std::function<float(float)> act)
{
	neurons_vect_[position].CalcValue(act);
}

//Creating a Layer with Neurons
Layer::Layer(Layerposition pos, int layer_size, bool do_add_bias, int prev_layer_size) {
	position_ = pos;
	layersize_ = layer_size;
	prev_layer_ = NULL;
	for (int i = 0; i < layersize_; i++)
	{
		//Every Neuron will create a Synapse vector of prev_layer_size elements
		neurons_vect_.push_back( Neuron(i, prev_layer_size ));
	}
	if (do_add_bias) {
		neurons_vect_.push_back(Neuron(layersize_,0));
		neurons_vect_[layersize_].MakeBias();
	}
}

// Sets a pointer to the previous layer passing it to every Neuron in this Layer
// This can be done only after all Layers are added to the Layers Vector of NeuralNetwork obj
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
	for (auto& prev_layer_neuron : prev_layer_->neurons_vect_)  // For every Neuron of the previous Layer
	{
		float f_error = 0.0F;
		float f_gradient = 0.0F;
		float f_correction = 0.0F;
		for (auto& this_layer_neuron : neurons_vect_) // For every Neuron of this Layer
		{
			if (this_layer_neuron.IsBias()) {
				continue;
			}//Bias Neruron has no Synapse vector

			//Multiplying this Layer Neuron Error by Synapse Value of the Synapse connecting this Layer Neuron with the Previous Layer Neuron
			f_error += this_layer_neuron.GetError() * this_layer_neuron.GetSynapsValForNeuron(prev_layer_neuron.GetIndex());
		}

		//Previous Layer Neuron Error = f_error * Activation Derivate func of Previous Neuron Layer Value
		prev_layer_neuron.SetError(f_error * act_deriv(prev_layer_neuron.GetValue()));
		
		//Immediately updating all Synapses connecting previos Layer Neuron with every Neuron of this Layer 
		for (auto& this_layer_neuron : neurons_vect_)
		{
			if (this_layer_neuron.IsBias()) continue; //Has no Synapses

			//Gradient = This Layer Neuron Error * Previous Layer Neuron Value
			f_gradient = this_layer_neuron.GetError() * prev_layer_neuron.GetValue();
			// LR * Gr + Momentum * (Last Difference of Synapse connecting this Layer Neuron with Previous Layer Neuron)
			f_correction = Learn_rate * f_gradient + Momentum * this_layer_neuron.GetSynapsDiffForNeuron(prev_layer_neuron.GetIndex());
			//Add f_correction to the Value of the Synapse connecting this Layer Neuron with Previous Layer Neuron
			//Refreshing the last diefference with f_correction
			this_layer_neuron.CorrectSynapsForNeuron(prev_layer_neuron.GetIndex(), f_correction);
		}
	}
}
void Layer::SaveNeuronsSynapses(std::ofstream& file)
{
	for (size_t i = 0; i < neurons_vect_.size(); i++)
	{
		neurons_vect_[i].WriteVectorToFile(file);
	}
}
int Layer::LayerSize() 
{
	return neurons_vect_.size()-1;
}