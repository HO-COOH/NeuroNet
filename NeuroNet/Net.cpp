#include "Net.h"
Net::Net(size_t numberOfLayers, size_t numberOfInputs): input(numberOfInputs,1)
{
	layers.reserve(numberOfLayers);
}

Net::Net(size_t numberOfLayers, size_t numberOfInputs, const Matrix& _input): input(_input)
{
	if (_input.row() != numberOfInputs)
	{
		std::cout << "Error in initialization of the Net. Using an input matrix of size: ";
		_input.reportSize();
		std::cout << " and the number of input is: " << numberOfInputs << std::endl;
	}
	layers.reserve(numberOfInputs);
}

void Net::init_input(const Matrix& _input)
{
	if (input.row() != _input.row())
	{
		std::cout << "The input matrix of size: ";
		_input.reportSize();
		std::cout << " can't init.\n";
	}
	input=_input;
	for (auto& each_layer : layers)
	{
		std::cout << each_layer.isOutput() << std::endl;
	}
}

void Net::init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases)
{
	//if (weights.size() != layers.size())
	//{
	//	std::cout << "The vector of weight matrcies size is " << weights.size() << " but there are " << layers.size() << " layers to init!\n";
	//	return;
	//}
	//if (biases.size() != layers.size())
	//{
	//	std::cout << "The vector of weight matrcies size is " << biases.size() << " but there are " << layers.size() << " layers to init!\n";
	//	return;
	//}
	size_t i = 0;
	for (auto& each_weight : weights)
	{
		layers.emplace_back(Layer(each_weight.row(), each_weight.column()));
	}
	layers[layers.size() - 1].setOutputFlag();
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
	}
	for (auto& each_layer : layers)
	{
		std::cout << each_layer.isOutput() << std::endl;
	}
}

//void Net::init_bias(const std::vector<Matrix> biases)
//{
//	if (biases.size() != layers.size())
//	{
//		std::cout << "The vector of weight matrcies size is " << biases.size() << " but there are " << layers.size() << " layers to init!\n";
//		return;
//	}
//	size_t layerIndex = 0;
//}

void Net::run(bool showResultOfEachLayer)
{
	for (auto& each_layer : layers)
	{
		std::cout << each_layer.isOutput() << std::endl;
	}
	if (!layers[layers.size() - 1].isOutput())
	{
		std::cout << "The last layer is not set in output mode!\n";
		return;
	}
	std::cout << "The input:\n" << input;
	if (!layers[0].isReady())
	{
		std::cout << "Layer 0 is not ready!\n";
		return;
	}
	layers[0].run(input);
	output = layers[0].output();
	if (showResultOfEachLayer)
		std::cout << "Layer 0:\n" << output;
	size_t layerIndex = 1;
	for (auto it = ++layers.begin(); it != layers.end(); ++it)
	{
		if (!it->isReady())
		{
			std::cout << "Layer " << layerIndex << " is not ready!\n";
			return;
		}
		it->run(output);
		output = it->output();
		if (showResultOfEachLayer)
			std::cout << "Layer " << layerIndex << ":\n" << output;
		++layerIndex;
	}
}
