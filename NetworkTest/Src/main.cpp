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
	
	//Creating and loading a dataset from the file
	NNData Icons;
	Icons.Read("C:/Users/�������/source/repos/Neural/x64/Debug/Folder_train/optdigits.tra");
	// Preparing dataset Setting values to fit in (-1:1)
	Icons.Prepare();
	
	std::vector<int> NetworkConfig = {48,50,50,10};

	NeuralNetwork Network;
	Network
		.SetLearnRate(0.7f)
		.SetMomentum(0.6f)
		.AddBias(false)
		.SetLayers(NetworkConfig);

	srand(time(NULL));
	for (size_t i = 1; i < NetworkConfig.size(); i++)
	{
		Network.InitSynapseInLayer(i, [&]()->float {return( static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * sqrt(2.0f / NetworkConfig[i - 1])); });
	}
	
	// Activation : [](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); }
	// Derivate : [](float val)->float {return val * (1.0F - val); }

}