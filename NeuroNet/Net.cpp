#include "Net.h"
#include <cmath>
#include <algorithm>	//for shuffle the inputs and desired_outputs pair
#include <random>
#include <chrono>
double tanh_prime(double x)
{
	return 1.0 - (tanh(x) * tanh(x));
}

Net::Net(size_t numberOfInputs, const std::vector<unsigned int> numberOfNeuronsEachLayer):numberOfInputs(numberOfInputs),numberOfLayers(numberOfNeuronsEachLayer.size())
{
	layers.push_back(Layer(numberOfNeuronsEachLayer[0], numberOfInputs));
	for (size_t index = 1; index < numberOfNeuronsEachLayer.size(); ++index)
	{
		layers.push_back(Layer(numberOfNeuronsEachLayer[index], layers.back().numberOfPerceptrons));
	}
	local_gradients.resize(numberOfNeuronsEachLayer.size());
}

Matrix Net::init_input(const Matrix& _input)
{
	if (_input.row()!= numberOfInputs)
	{
		std::cout << "The input matrix of size: ";
		_input.reportSize();
		std::cout << " can't init.\n";
	}
	Matrix temp(_input.row() + 1, 1);
	temp(1, 1) = 1;
	for (size_t rowIndex = 1; rowIndex <= numberOfInputs; ++rowIndex)
	{
		temp(rowIndex + 1, 1) = _input(rowIndex, 1);
	}
	return temp;
}

void Net::init_inputs(const std::vector<Matrix>& _inputs)
{
	for (auto& each_input : _inputs)
		inputs.push_back(init_input(each_input));
}

void Net::init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases)
{
	if (weights.size() != numberOfLayers)
	{
		std::cout << "The vector of weight matrcies size is " << weights.size() << " but there are " << numberOfLayers << " layers to init!\n";
		return;
	}
	if (biases.size() != numberOfLayers)
	{
		std::cout << "The vector of weight matrcies size is " << biases.size() << " but there are " << numberOfLayers << " layers to init!\n";
		return;
	}
	size_t i = 0;
	layers.back().setOutputFlag();
	for (size_t layerIndex=0; layerIndex<layers.size();++layerIndex)
	{
		if (weights[layerIndex].row() != layers[layerIndex].numberOfPerceptrons || weights[layerIndex].column() != layers[layerIndex].numberOfConnections||biases[layerIndex].row()!=layers[layerIndex].numberOfPerceptrons||biases[layerIndex].column()!=1)
		{
			std::cout << "The " << layerIndex << " -th weight matrix to init is of wrong size: ";
			weights[layerIndex].reportSize();
			std::cout << std::endl << "But the layer: "; 
			layers[layerIndex].report();
			return;
		}
		layers[layerIndex].set(weights[layerIndex], biases[layerIndex]);
		local_gradients[layerIndex].resize(weights[layerIndex].row(), 1);
	}
}

void Net::set_desired_outputs(const std::vector<Matrix> _desired_outputs)
{
	desired_outputs = _desired_outputs;
}

Matrix Net::ForwardComputation(size_t k)
{
	if (desired_outputs.size() != inputs.size())
	{
		std::cout << "The batch only contains" << desired_outputs.size() << " desired outputs but there are " << inputs.size() << " inputs!\n";
		abort();
	}
	if (!layers[layers.size() - 1].isOutput())
	{
		std::cout << "The last layer is not set to output mode!\n";
		abort();
	}
	layers[0].run(inputs[k]);
	Matrix out0 = layers[0].output();
	size_t layerIndex = 0;
	for (size_t layerIndex=1;layerIndex<layers.size();++layerIndex)
	{
		if (!layers[layerIndex].isReady())
		{
			std::cout << "Layer" << layerIndex << " in " << k << " iteration is not ready!\n";
			abort();
		}
		else
		{
			layers[layerIndex].run(out0);
			out0 = layers[layerIndex].output();
		}
	}
	return out0;
}

Matrix Net::getError(size_t k)
{
	using namespace std;
	if (desired_outputs[k].row() != layers[numberOfLayers - 1].numberOfPerceptrons|| desired_outputs[k].column()!=1)
	{
		cout << "The size of desired Output is wrong!\n";
		desired_outputs[k].reportSize();
		cout << "There are " << layers[numberOfLayers - 1].numberOfPerceptrons << " neurons in the output layer!\n";
	}
	Matrix error(desired_outputs[k].row(), 1);
	for (size_t i = 0; i < desired_outputs[k].row(); ++i)
	{
		error(i + 1, 1) = desired_outputs[k](i + 1, 1) - layers[layers.size()-1].output()(i+2,1);
	}
	return error;
}



void Net::BackwardComputation(size_t k)
{
	Matrix error = getError(k);
	local_gradients[numberOfLayers - 1] = error;
	//layer (0-numberofLayers-2)
	for (int layerIndex = numberOfLayers - 2; layerIndex >= 0; --layerIndex)
	{
		// (tanh x)'=1-(tanh x)^2
		// neurons 0-numberOfPerceptrons
		for (size_t rowIndex = 0; rowIndex < layers[layerIndex].numberOfPerceptrons; ++rowIndex)
		{
			double sum = 0;
			//sum up the gradient from back layer
			for (size_t i = 0; i < layers[layerIndex + 1].numberOfPerceptrons; ++i)
				sum += local_gradients[layerIndex + 1](i + 1, 1) * layers[layerIndex + 1].get_weight()(i + 1, rowIndex+2);
			local_gradients[layerIndex](rowIndex + 1, 1) = tanh_prime(layers[layerIndex].get_neuron_origin_sum(rowIndex)) * sum;
		}
	}
	/*We have local_gradients, now compute the updated weights*/
		//for layer 0 - numberofLayer-1
	for (size_t layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
	{
		//Matrix updatedWeight;
		//if (layerIndex == 0)
		//	updatedWeight = layers[layerIndex].get_weight() + ((local_gradients[layerIndex] * inputs[k].transpose()) * learning_rate);
		//else
		//	updatedWeight = layers[layerIndex].get_weight() + ((local_gradients[layerIndex] * layers[layerIndex-1].output().transpose()) * learning_rate);
		//layers[layerIndex].set(updatedWeight);
		Matrix updatedWeight(layers[layerIndex].numberOfPerceptrons, layers[layerIndex].numberOfConnections + 1);
		if (layerIndex == 0)
		{
			for (size_t j = 0; j < updatedWeight.row(); ++j)
			{
				for (size_t i = 0; i < updatedWeight.column(); ++i)
				{
					updatedWeight(j + 1, i + 1) = layers[layerIndex].get_weight()(j + 1, i + 1) + learning_rate * local_gradients[layerIndex](j + 1, 1) * inputs[k](i + 1, 1);
				}
			}
		}
		else
		{
			for (size_t j = 0; j < updatedWeight.row(); ++j)
			{
				for (size_t i = 0; i < updatedWeight.column(); ++i)
				{
					updatedWeight(j + 1, i + 1) = layers[layerIndex].get_weight()(j + 1, i + 1) + learning_rate * local_gradients[layerIndex](j + 1, 1) * layers[layerIndex-1].output()(i + 1, 1);
				}
			}
		}
		layers[layerIndex].set(updatedWeight);
	}
}

void Net::run(size_t iterations, bool showEachPass)
{
	using namespace std;
	default_random_engine rd;
	uniform_int_distribution<unsigned> random_index(0, inputs.size() - 1);
	for (size_t times = 0; times < iterations; times++)
	{
		unsigned i = random_index(rd);
		if (showEachPass)
		{
			cout << "Pass " << times << ":\n";
			cout << "Inputs:\n" << inputs[i];
			cout << "Outputs:\n" << ForwardComputation(i);
			cout << "Target:\n" << desired_outputs[i];
		}
		else
			ForwardComputation(i);
		BackwardComputation(i);
	}
	cout << "final weight:\n\n";
	for (auto& each_layer : layers)
	{
		each_layer.ShowWeight();
		cout << endl;
	}
}

//void Net::update_weights()
//{
//	//for layer 0 - numberofLayer-1
//	for (size_t layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
//	{
//		Matrix updatedWeight(layers[layerIndex].numberOfPerceptrons, layers[layerIndex].numberOfConnections + 1);
//		//for each neuron 
//		for (size_t neuronIndex = 0; neuronIndex < layers[layerIndex].numberOfPerceptrons; ++neuronIndex)
//		{
//			//for each connection to the layer
//			for (size_t connectionIndex = 0; connectionIndex < layers[layerIndex].numberOfConnections+1; ++connectionIndex)
//			{
//				if (connectionIndex == 0)
//					updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * 1;
//				else
//				{
//					if (layerIndex == 0)
//						updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * input(connectionIndex + 1, 1);
//					else
//						updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * layers[layerIndex - 1].get_neuron_output(connectionIndex-1);
//				}
//			}
//		}
//		layers[layerIndex].set(updatedWeight);
//	}
//}


//////////////////////////////////////////Debug//////////////////////////////////////////
void Net::show_all_weights()
{
	for (auto& each_layer : layers)
	{
		each_layer.ShowWeight();
		std::cout << std::endl;
	}
}

void Net::show_local_gradients()
{
	using namespace std;
	for (auto& each_local_gradient : local_gradients)
	{
		cout << each_local_gradient << endl;
	}
}

void Net::show_layers()
{
	for (auto& each_layer : layers)
	{
		each_layer.report();
		std::cout << std::endl;
	}
}

void Net::show_input_desired_output_pair()
{
	using namespace std;
	if (inputs.size() != desired_outputs.size())
	{
		cout << "Inputs.size()!=desired_outputs.size()\n";
		return;
	}
	for (size_t index = 0; index < inputs.size(); ++index)
	{
		cout << index << ":\n";
		cout << inputs[index] << endl << desired_outputs[index] << endl;
	}
}

void Net::test(size_t input_index)
{
	using namespace std;
	cout << "\nTarget outputs:\n" << desired_outputs[input_index] << "Neural Net output:\n";
	cout << ForwardComputation(input_index).slice(2,layers[layers.size()-1].numberOfPerceptrons+1,1,1);
}


