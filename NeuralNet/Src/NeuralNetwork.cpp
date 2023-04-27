#include "NeuralNetwork.h"
#include "Neuron.h"
#include "string"
#include <iostream>
#include<iomanip>
#include <fstream>
#include "Neuron.h"
#include "Layer.h"
#include <iterator>
#include <sstream>

struct NeuralNetwork::Impl
{
	Impl();
	Impl(INeuralNetworkParametres& params);
	void ImplSetLayers(const std::vector<int>& Configuration);

	float Learn_rate;
	float Momentum;
	bool Do_add_bias;
	bool Isinitialized = false;
	std::vector<Layer> Layer_vect_;
	std::vector<int> Network_configuration;
	std::vector<float> OutputDifference;
};

NeuralNetwork::Impl::Impl() :
Learn_rate(0.0F), Momentum(0.0f), Do_add_bias(false), Isinitialized(false)
{}

NeuralNetwork::~NeuralNetwork() {}

NeuralNetwork::Impl::Impl(INeuralNetworkParametres& params):
Learn_rate(params.SetNeuralNetworkLearnRate()),
Momentum(params.SetNeuralNetworkMomentum()),
Do_add_bias(params.DoAddBias())
{
	// Initializing the Neywork_configuration vectot with layers sizes  
	params.SetNeuralNetworkConfiguration(Network_configuration);
	//Creating Layers of Neurons with Synapses 
	ImplSetLayers(Network_configuration);

}

void NeuralNetwork::Impl::ImplSetLayers(const std::vector<int>& Configuration)
{
	if (Isinitialized == true)
		throw std::invalid_argument("Already initialyzed!");


	for (size_t i = 0; i < Configuration.size(); i++) //for every layer
	{
		Network_configuration.push_back(Configuration[i]);
		if (i == 0)
			Layer_vect_.push_back(Layer(Layerposition::Input, Configuration[i], Do_add_bias, 0));
		else if (i == Network_configuration.size() - 1)
			Layer_vect_.push_back(Layer(Layerposition::Output, Configuration[i], Do_add_bias, Layer_vect_[i - 1].NeuronsCount()));
		else
			Layer_vect_.push_back(Layer(Layerposition::Hidden, Configuration[i], Do_add_bias, Layer_vect_[i - 1].NeuronsCount()));
	}

	for (size_t i = 1; i < Configuration.size(); i++)
	{
		Layer_vect_[i].SetPreviousLayerPtr(&(Layer_vect_[i - 1]));
	}


	for (size_t i = 0; i < Configuration[Configuration.size() - 1]; i++)
	{
		OutputDifference.push_back(0.0F);
	}

}


//Creates a new empty NeuralNetwork Obj without Layers
NeuralNetwork::NeuralNetwork() :
	pimpl( new Impl())
{
}


//Creates Neural Network from INeuralNetworkParametres object
NeuralNetwork::NeuralNetwork(INeuralNetworkParametres& params) :
	pimpl( new Impl (params))
{
}

NeuralNetwork& NeuralNetwork::SetMomentum(float momentum) { pimpl->Momentum = momentum; return *this; }
NeuralNetwork& NeuralNetwork::SetLearnRate(float rate) { pimpl->Learn_rate = rate; return *this; }


NeuralNetwork& NeuralNetwork::AddBias(bool bias) 
{
	if (pimpl->Isinitialized) 
		throw std::invalid_argument("Neural Network already initialyzed! Please decide on Bias before calling SetLayers method!");

	pimpl->Do_add_bias = bias;

	return *this; 
}

void NeuralNetwork::InitSynapseInLayer(int layernum, std::function<float()> initfunc) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	pimpl->Layer_vect_[layernum].InitSynapse(initfunc);
}
void NeuralNetwork::SetInputNeuron(int position, float value) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	float value_to_set = value <= 1.0F and value >= -1.0F ? value : throw std::invalid_argument("Ivalid neuron value!");
	int set_position = position >= 0 and position < pimpl->Network_configuration[0] ? position : throw std::invalid_argument("Invalid synapse postion!");
	pimpl->Layer_vect_[0].SetNeuronValue(set_position, value_to_set);
}

void NeuralNetwork::Calculate(std::function<float(float)> act) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");


	for (size_t i = 1; i < pimpl->Layer_vect_.size(); i++)
	{
		for (int j = 0; j < pimpl->Network_configuration[i] ; j++)
		{			
			pimpl->Layer_vect_[i].CalculateNeuron(j, act);
		}
		
	}
}

//Calculates the difference between the output layer Neuron Value and provided true value
//The difference will be stored in OutputDifference vector for fufture MSE calculation
//Sets the output Neuron Error = difference * Derivate Activate func 
void NeuralNetwork::CalcOutputError(int position, float true_value, std::function<float(float)> act_deriv)
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");
    
    Layer &output_layer = pimpl->Layer_vect_[pimpl->Layer_vect_.size()-1];
	int _position = (position >= 0) and (position < output_layer.NeuronsCount()) ? position : throw std::invalid_argument("Invalid neuron postion!");

	pimpl->OutputDifference[_position] = true_value - output_layer.GetNeuronVal(_position);   //Saving difference for MSE
	
	float Error = (true_value - output_layer.GetNeuronVal(_position)) *  act_deriv(output_layer.GetNeuronVal(_position));
	output_layer.SetNeuronError(_position, Error); //�����

}

void NeuralNetwork::BackPropogationError(std::function<float(float)> act_deriv) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	for (size_t i = pimpl->Layer_vect_.size()-1; i > 0; i--) //For every Layer starting from the last one down to the first hidden Layer
	{
		pimpl->Layer_vect_[i].BackPropogate(act_deriv, pimpl->Learn_rate, pimpl->Momentum);
	}
}
float NeuralNetwork::MeanSquaredError()
{	
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	float MSE = 0.0F;
	for (size_t i = 0; i < pimpl->OutputDifference.size(); i++)
	{
		MSE += pimpl->OutputDifference[i] * pimpl->OutputDifference[i];
	}
	MSE /= pimpl->OutputDifference.size();
	return MSE;
}

void NeuralNetwork::ShowNeuralNetworkParametres(std::ostream &output) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	output << "Neural network has the following structure: \n";
	output << "Amount of layers:"<< pimpl->Network_configuration.size()<<std::endl;
	for (size_t i = 0; i < pimpl->Network_configuration.size(); i++)
	{
		output << "Layer " << i << " has " << pimpl->Network_configuration[i] << " neurons\n";

	}
	if (pimpl->Do_add_bias) {
		output << "Network uses bias";
	}
	else {
		output<< "Network doesnt use bias";
	} 
	output << std::endl;
	output << "Momentum = " << pimpl->Momentum << std::endl;
	output << "Learn rate = " << pimpl->Learn_rate << std::endl;
}

float NeuralNetwork::GetOutputNeuronValue(int neuron_index) 
{
	if (!pimpl->Isinitialized)
		throw std::invalid_argument("Neural Network was not initialyzed!");

	if( neuron_index > pimpl->Layer_vect_[pimpl->Layer_vect_.size() - 1].NeuronsCount()) throw  std::invalid_argument("Neuron index out of range!");
	return pimpl->Layer_vect_[pimpl->Layer_vect_.size() - 1].GetNeuronVal(neuron_index);
}

//Creating layers of neurons with synapses
NeuralNetwork& NeuralNetwork::SetLayers(const std::vector<int>& Configuration ) 
{
	pimpl->ImplSetLayers(Configuration);
	pimpl->Isinitialized = true;
	return *this;	
}

/*
void NeuralNetwork::SetSynapseByIndex(int layernum, int neuronnum, int synapse_index, float value)
{
	Layer_vect_[layernum].neurons_vect_[neuronnum].synaps_vect[synapse_index].f_value_ = value;
}
*/


void NeuralNetwork::Save(std::string file_name)
{
	std::ofstream file(file_name);
	file << pimpl->Momentum<<std::endl;
	file << pimpl->Learn_rate<<std::endl;
	file << pimpl->Layer_vect_.size() << std::endl;
	for (size_t i = 0; i < pimpl->Layer_vect_.size(); i++)
	{
		file << pimpl->Layer_vect_[i].LayerSize()<<" ";

	}
	file << std::endl;
	for (size_t i = 1; i < pimpl->Layer_vect_.size(); i++)
	{
		pimpl->Layer_vect_[i].SaveNeuronsSynapses(file);
		file << std::endl;
	}
	file.close();
}
void  NeuralNetwork::Load(std::string file_name) 
{
	std::ifstream file(file_name);
	std::string temp;
	std::getline(file, temp);
	pimpl->Momentum = ::atof(temp.c_str()); //присвоили моментум

	std::getline(file, temp);
	pimpl->Learn_rate = ::atof(temp.c_str()); //присвоили скорость обучения

	std::getline(file, temp);
	int layeramount = ::atof(temp.c_str()); // получили количество слоев

	std::getline(file, temp);
	std::vector<int> v;
	std::stringstream ss(temp);
	std::copy(std::istream_iterator<int>(ss), {}, back_inserter(v));
	SetLayers(v);
	
	std::getline(file, temp);
	std::vector<double> temp_vect;
	std::stringstream str(temp);
	std::copy(std::istream_iterator<double>(str), {}, back_inserter(temp_vect)); //Вылетает тут, надо разбирать...
	for (size_t i = 0; i < pimpl->Layer_vect_.size(); i++)
	{
		for (size_t j = 0; j < v[i]; j++)
		{
			for (size_t k = 0; k < v[i+1]; k++)
			{
				pimpl->Layer_vect_[i].SetNeuronValue(j, temp_vect[k]);
			}
			
		}
		
	}
	
}

	

