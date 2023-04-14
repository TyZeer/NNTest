#include "NeuralNetwork.h"
#include "NeuralNetworkParametres.h"
#include "NNData.h"
#include <time.h>
#include <math.h>
#include <vector>
#include <iomanip>
#include <functional>
#include <cmath>

#define  _USE_MATH_DEFINES
#define INPUT_LAYER_SIZE 64
#define OUTPUT_LAYER_SIZE 10


    
void RunNewSet( NNIcon& Icon, NeuralNetwork& Network );
int GetResult( NeuralNetwork& Network, int output_layer_size );
void CalcOutputError( NNIcon & Icon, NeuralNetwork& Network, int output_layer_size );

int main()
{
    
    //Creating and loading a dataset from the file
    NNData Icons;
    Icons.Read("C:/Users/ִלטענטי/Desktop/Dataset/optdigits.tra");
    // Preparing dataset Setting values to fit in (-1:1)
    Icons.Prepare();
    
    std::vector<int> NetworkConfig = {64,50,50,10};

    NeuralNetwork Network;
    Network
        .SetLearnRate(0.7f)
        .SetMomentum(0.6f)
        .AddBias(true)
        .SetLayers(NetworkConfig);

    srand((unsigned)time(NULL));
    for (int i = 1; i < NetworkConfig.size(); i++)
    {
        Network.InitSynapseInLayer(i, [&]()->float {return( static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * sqrt(2.0f / NetworkConfig[i - 1])); });
    }
    
    // Activation : [](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); }
    // Derivate : [](float val)->float {return val * (1.0F - val); }
    
    Network.ShowNeuralNetworkParametres(std::cout);
    std::cout << Icons.icon_vect_.size() << " icons loaded from optdigits.tra \n";
    
    for( int times = 0; times < 3; times ++){
        std::cout << "\nRunning set # " << times+1 << std::endl;
        int rights = 0;
        int set_no = 0, sub_count = 0;
        for(auto& Icon : Icons.icon_vect_){
            RunNewSet(Icon, Network);
            if( GetResult(Network, NetworkConfig[NetworkConfig.size()-1]) == Icon.value_)
                rights++;
            
            CalcOutputError(Icon, Network,  NetworkConfig[NetworkConfig.size()-1]);
            
            if(sub_count == 100){
                std::cout << set_no << " icons processed with " << rights << "% success \n";
                sub_count = 0;
                rights = 0;
            }
            
            sub_count++;
            set_no++;
            
            Network.BackPropogationError([](float val)->float {return val * (1.0F - val); });
        }
    }
}



void RunNewSet( NNIcon& Icon, NeuralNetwork& Network )
{
    int i = 0;
    for(float value : Icon.data_){
        Network.SetInputNeuron(i++, value);
    }
    Network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(exp(1), val * -1.0f)); });
}

int GetResult( NeuralNetwork& Network, int output_layer_size )
{
    float max_result = 0.0f;
    int max_neuron_index = 0;
    
    for( int i = 0; i < output_layer_size; i++)
    {
        if( max_result < Network.GetOutputNeuronValue(i))
        {
            max_result = Network.GetOutputNeuronValue(i);
            max_neuron_index = i;
        }
    }
    return max_neuron_index;
}

void CalcOutputError( NNIcon & Icon, NeuralNetwork& Network, int output_layer_size )
{
    for( int i = 0; i < output_layer_size; i++)
    {
        if( i == Icon.value_)
            Network.CalcOutputError(i, 0.99f, [](float val)->float {return val * (1.0F - val); });
        else
            Network.CalcOutputError(i, 0.0f, [](float val)->float {return val * (1.0F - val); });
    }
}
