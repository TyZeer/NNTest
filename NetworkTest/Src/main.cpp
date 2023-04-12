#include "NeuralNetwork.h"
#include "NeuralNetworkParametres.h"
#include "NNData.h" 
#include <time.h>
#include <math.h>
#include <vector>
#include<iomanip>

#define INPUT_LAYER_SIZE 64
#define OUTPUT_LAYER_SIZE 10
	

int main()
{
	NNParametres config;
	NeuralNetwork network(config);
	network.ShowNeuralNetworkParametres(std::cout);
	NNData Dataset;
	Dataset.Read("C:/Users/Дмитрий/source/repos/Neural/x64/Debug/Folder_train/optdigits.tra");
	Dataset.Prepare();
	srand(time(NULL));
	std::vector<int > neur_vect{ 2,2,1 };
	
	network.SetSynapseByIndex(1, 0, 0, 0.45F); //W1
	network.SetSynapseByIndex(1, 0, 1, -0.12F); //W3
	network.SetSynapseByIndex(1, 1, 0, 0.78F);  //W2
	network.SetSynapseByIndex(1, 1, 1, 0.13F); //W4
	network.SetSynapseByIndex(2, 0, 0, 1.5F); //W5
	network.SetSynapseByIndex(2, 0, 1, -2.3F); //W5

	network.SetInputNeuron(0, 1.0F); //I1
	network.SetInputNeuron(1, 0.0F); //I2

	network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); });
	network.CalcOutputError(0, 1.0f, [](float val)->float {return val * (1.0F - val); });
	//network.MediumSquareError();
	network.BackPropogationError([](float val)->float {return val * (1.0F - val); });
	

	for (size_t i = 1; i < 4; i++)
	{
		network.InitSynapseInLayer(i, [&]()->float {return( static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * sqrt(2.0f / neur_vect[i - 1])); });
	}

	/*
	for (size_t i = 0; i < 64; i++)
	{
		network.SetInputNeuron(i, Dataset.icon_vect_[0].data_[i]);
	}
	network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); });
	for (int i = 0; i < 10; i++)
	{
		if (i == Dataset.icon_vect_[i].value_)
			network.CalcOutputError(i, 1.0f, [](float val)->float {return val * (1.0F - val); });
		else
			network.CalcOutputError(i, 0.0f, [](float val)->float {return val * (1.0F - val); });
	}
	*/

	//int Guess_count = 0;
	//float MaxNeuronVal = 0.0F;
	//int CurrentMaxIndex = 0;

	//for (size_t i = 0; i < 20 /*Dataset.icon_vect_.size()*/; i++)
	//{
	//	for (size_t j = 0; j < INPUT_LAYER_SIZE; j++)
	//	{
	//		network.SetInputNeuron(j, Dataset.icon_vect_[0].data_[j]); //Берем одну и ту же иконку
	//	}
	//	network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); });

	//	MaxNeuronVal = 0.0F;
	//	CurrentMaxIndex = 0;
	//	for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
	//	{
	//		if (network.GetOutputNeuronValue(i) > MaxNeuronVal) {
	//			MaxNeuronVal = network.GetOutputNeuronValue(i);
	//			CurrentMaxIndex = i;
	//		}
	//	}
	//	if (CurrentMaxIndex == Dataset.icon_vect_[i].value_) Guess_count++;

	//	for (int k = 0; k < OUTPUT_LAYER_SIZE; k++)
	//	{
	//		if (k == Dataset.icon_vect_[0].value_)
	//			network.CalcOutputError(k, 1.0f, [](float val)->float {return val * (1.0F - val); });
	//		else
	//			network.CalcOutputError(k, 0.0f, [](float val)->float {return val * (1.0F - val); });

	//	}
	//	if (i % 1 == 0) {
	//		std::cout << "MSE = " << std::setprecision(6) << network.MediumSquareError() << std::endl;
	//		std::cout << "guessed " << Guess_count << "% of icons \n";
	//		Guess_count = 0;
	//	}
	//	network.BackPropogationError([](float val)->float {return val * (1.0F - val); });

	//}
	

	//network.Save("C:/Users/Дмитрий/Desktop");

}