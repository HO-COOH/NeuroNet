#pragma once
#include <vector>
#include "Layer.h"

class Net
{
	std::vector<Layer> layers;
	Matrix input;
	
public:
	Net(size_t numberOfLayers, size_t numberOfInputs);
	Net(size_t numberOfLayers, size_t numberOfInputs, const Matrix& _input);

	void init_input(const Matrix& _input);
	void init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases);
	//void init_bias(const std::vector<Matrix> biases);

	void run(bool showResultOfEachLayer=false);

	Matrix output;
};

