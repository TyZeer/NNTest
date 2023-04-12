#include "NeuralNetwork.h"
#include "Neuron.h"
#include "string"
#include<iomanip>
#include <fstream>

NeuralNetwork::NeuralNetwork() {
	if (Isinitialized == true) {
		throw std::invalid_argument("Already initialyzed!");
	}
	else {
		std::vector<int> v{ 0 };
		Learn_rate = 0.0f;
		Momentum = 0.0f;
		Do_add_bias = false;
		SetLayer(v);
		Isinitialized = true;
	}
}
void NeuralNetwork::SetMomentum(float moment) {
	Momentum = moment;
}

NeuralNetwork::NeuralNetwork(INeuralNetworkParametres& params) :

	Learn_rate(params.SetNeuralNetworkLearnRate()), 
	Momentum(params.SetNeuralNetworkMomentum()),
	Do_add_bias(params.DoAddBias())

	
{
	if (Isinitialized == true) {
		throw std::invalid_argument("Already initialyzed!");
	}
	else {
		params.SetNeuralNetworkConfiguration(Network_configuration);

		//Layer_vect_.resize(Network_configuration.size());

		//Layer* layer_ptr;
		SetLayer(Network_configuration);
		Isinitialized = true;
	}
}

void NeuralNetwork::InitSynapseInLayer(int layernum, std::function<float()> initfunc) {

	Layer_vect_[layernum].InitSynapse(initfunc);
	
}
void NeuralNetwork::SetInputNeuron(int position, float value) {

	float value_to_set = value <= 1.0F and value >= -1.0F ? value : throw std::invalid_argument("Ivalid neuron value!");
	int set_position = position >= 0 and position < Network_configuration[0] ? position : throw std::invalid_argument("Invalid synapse postion!");
	Layer_vect_[0].SetNeuronValue(set_position, value_to_set);
}
void NeuralNetwork::Calculate(std::function<float(float)> act) {
	for (size_t i = 1; i < Layer_vect_.size(); i++)
	{
		for (int j = 0; j < Network_configuration[i] ; j++) 
		{
			//Layer_vect_[i].GetNeuron(j)->CalcValue(act); //тут менял
			Layer_vect_[i].CalculateNeuron(j, act);

		}
		
	}
}
void NeuralNetwork::CalcOutputError(int position, float true_value, std::function<float(float)> act_deriv)
{
	int set_position = position >= 0 and position < Network_configuration[0] ? position : throw std::invalid_argument("Invalid synapse postion!");

	float Error=true_value - Layer_vect_[Layer_vect_.size()-1].GetNeuronVal(set_position);   //тут менял
	OutputDifference[position] = Error;
	Error *= act_deriv(Error);
	Layer_vect_[Layer_vect_.size() - 1].SetNeuronError(set_position, Error); //менял

}
void NeuralNetwork::BackPropogationError(std::function<float(float)> act_deriv) 
{
	for (size_t i = Layer_vect_.size()-1; i > 0; i--)
	{
		Layer_vect_[i].BackPropogate(act_deriv, Learn_rate, Momentum);
	}
}
float NeuralNetwork::MediumSquareError()
{	
	float MSE = 0.0F;
	for (size_t i = 0; i < OutputDifference.size(); i++)
	{
		MSE += OutputDifference[i] * OutputDifference[i];
	}
	MSE /= OutputDifference.size();
	return MSE;
}
void NeuralNetwork::ShowNeuralNetworkParametres(std::ostream &output) 
{
	output << "Neural network has the following structure: \n";
	output << "Amount of layers:"<<Network_configuration.size()<<std::endl;
	for (size_t i = 0; i < Network_configuration.size(); i++)
	{
		output << "Layer " << i << " has " << Network_configuration[i] << " neurons\n";

	}
	if (Do_add_bias) {
		output << "Network uses bias";
	}
	else {
		output<< "Network doesnt use bias";
	} 
	output << std::endl;
	output << "Momentum = " << Momentum << std::endl;
	output << "Learn rate = " << Learn_rate << std::endl;
}

float NeuralNetwork::GetOutputNeuronValue(int neuron_index) 
{
	if( neuron_index > Layer_vect_[Layer_vect_.size() - 1].NeuronsCount()) throw  std::invalid_argument("Neuron index out of range!");
	return Layer_vect_[Layer_vect_.size() - 1].GetNeuronVal(neuron_index);
}

void NeuralNetwork::SetLayer(std::vector<int> &Configuration) {
	if (Isinitialized == true) {
		throw std::invalid_argument("Already initialyzed!");
	}
	else {
		for (size_t i = 0; i < Configuration.size(); i++)
		{
			if (i == 0)
				Layer_vect_.push_back(Layer(Layerposition::Input, Configuration[i], Do_add_bias, 0));
			else if (i == Network_configuration.size() - 1)
				Layer_vect_.push_back(Layer(Layerposition::Output, Configuration[i], Do_add_bias, Layer_vect_[i - 1].GetTrueSize()));
			else
				Layer_vect_.push_back(Layer(Layerposition::Hidden, Configuration[i], Do_add_bias, Layer_vect_[i - 1].GetTrueSize()));

		}
		for (size_t i = 1; i < Configuration.size(); i++)
		{
			Layer_vect_[i].SetPreviousLayerPtr(&(Layer_vect_[i - 1]));
		}


		for (size_t i = 0; i < Configuration[Configuration.size() - 1]; i++)
		{
			OutputDifference.push_back(0.0F);
		}
		Isinitialized = true;
	}
	
}

void NeuralNetwork::SetSynapseByIndex(int layernum, int neuronnum, int synapse_index, float value)
{
	Layer_vect_[layernum].neurons_vect_[neuronnum].synaps_vect[synapse_index].f_value_ = value;
}

/*
void NeuralNetwork::Save(std::string path) 
{
	if (Layer_vect_.size() == 0) { throw std::invalid_argument("Has zero layers!"); }
	std::fstream ofs(path);
	float SynapseVal;
	for (int i = 0; i < Network_configuration.size(); i++)
	{
		for (size_t i = 0; i < length; i++)
		{

		}


	}
}
void NeuralNetwork::Load(std::string path) {}
*/