#pragma once
#include "Neuron.h"
#include "Matrix.h"
#include <vector>

class Layer
{
	std::vector<Neuron> perceptrons;
	Matrix weight;	//weight is (number of perceptrons, number of connections+1) size Matrix
	bool outputFlag = false;
public:
	Layer():numberOfConnections(0), numberOfPerceptrons(0) {}
	explicit Layer(size_t _numberOfPerceptrons, size_t _numberOfConnections);
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
	Matrix output() const;

	size_t numberOfConnections;
	size_t numberOfPerceptrons;

	void report() const;
	//Matrix get();
	bool isReady() const;

	/*BackProp functions*/
	Matrix& get_weight()
	{
		return weight;
	}
	double get_neuron_origin_sum(size_t index)
	{
		return perceptrons[index].sum;
	}
	double get_neuron_output(size_t index)
	{
		return perceptrons[index].get();
	}

	/*Debug*/
	void ShowWeight() const
	{
		std::cout << weight;
	}
};

