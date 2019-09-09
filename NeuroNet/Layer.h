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
	bool outputFlag = false;
public:
	Layer():numberOfConnections(0), numberOfPerceptrons(0) {}
	explicit Layer(size_t _numberOfPerceptrons, size_t _numberOfConnections);
	Layer(const Matrix& _weight, const Matrix& _bias);
	Layer(const Matrix& _weight);

	/*set and run*/
	void set(const Matrix& weight, const Matrix& bias);
	void set(const Matrix& weight);
	void setOutputFlag()
	{
		outputFlag = true;
	}
	bool isOutput()
	{
		return outputFlag;
	}
	void run(const Matrix& lastLayerOutput);
	Matrix output();

	size_t numberOfConnections;
	size_t numberOfPerceptrons;
	void report();
	//Matrix get();
	bool isReady() const;
};

