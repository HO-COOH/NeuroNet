#pragma once
#include "Neuron.h"
#include "Matrix.h"
#include <vector>

struct Perceptron
{
	Matrix weight;
	Neuron neuron;
};

class Layer
{
	std::vector<Perceptron> perceptrons;
public:
	explicit Layer(size_t numberOfPerceptrons);
	void set(const Matrix& weight);
	Matrix get();
};

