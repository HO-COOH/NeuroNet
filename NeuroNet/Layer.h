#pragma once
#include "Neuron.h"
#include "Matrix.h"
#include <vector>

struct Perceptron
{
	Matrix weight;
	double bias=0;
	Neuron neuron;
};

class Layer
{
	std::vector<Perceptron> perceptrons;
	Matrix weight;
	Matrix bias;
	size_t numberOfConnections;
	size_t numberOfPerceptrons;
public:
	explicit Layer(size_t _numberOfPerceptrons, size_t _numberOfConnections);
	Layer(const Matrix& _weight, const Matrix& _bias);
	Layer(const Matrix& _weight);

	/*set and run*/
	void set(const Matrix& weight, const Matrix& bias);
	void set(const Matrix& weight);
	void run(const Matrix& lastLayerOutput);
	Matrix output();


	void report();
	//Matrix get();
};

