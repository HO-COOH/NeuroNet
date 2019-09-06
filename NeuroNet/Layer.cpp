#include "Layer.h"
#include "Matrix.h"
#include <iostream>

Layer::Layer(size_t _numberOfPerceptrons, size_t _numberOfConnections):perceptrons(_numberOfPerceptrons), numberOfConnections(_numberOfConnections), numberOfPerceptrons(_numberOfPerceptrons)
{
	for (auto& x : perceptrons)
	{
		x.weight.resize(1, _numberOfConnections);
	}
}

Layer::Layer(const Matrix& _weight, const Matrix& _bias):weight(_weight),bias(_bias),numberOfConnections(_weight.column()),numberOfPerceptrons(_weight.row())
{
}

Layer::Layer(const Matrix& _weight):weight(_weight), bias(zeros(_weight.row())), numberOfConnections(_weight.column()), numberOfPerceptrons(_weight.row())
{
}

void Layer::set(const Matrix& weight, const Matrix& bias)
{
	if (weight.column() != numberOfConnections || weight.row() != numberOfPerceptrons || bias.row() != numberOfPerceptrons || bias.column() != 1)
	{
		std::cout << "Can't set this layer to a weight matrix of size: ";
		weight.reportSize();
		std::cout << " and a bias vector of size: ";
		bias.reportSize();
		std::cout << std::endl;
		return;
	}
	this->weight = weight;
	this->bias = bias;
}

void Layer::set(const Matrix& weight)
{
	if (weight.column() != numberOfConnections || weight.row() != numberOfPerceptrons)
	{
		std::cout << "Can't set this layer to a weight matrix of size: ";
		weight.reportSize();
		return;
	}
	this->weight = weight;
}

void Layer::run(const Matrix& lastLayerOutput)
{
	if (lastLayerOutput.row() != numberOfConnections || lastLayerOutput.column() != 1)
	{
		std::cout << "The size of last layer output matrix ";
		lastLayerOutput.reportSize();
		std::cout << " is wrong!\n";
		return;
	}
	Matrix result = weight * lastLayerOutput + bias;
	for (size_t rowIndex = 0; rowIndex < numberOfPerceptrons; ++rowIndex)
	{
		perceptrons[rowIndex].neuron.set(result(rowIndex + 1, 1));
	}
}

Matrix Layer::output()
{
	Matrix output(numberOfPerceptrons, 1);
	for (size_t rowIndex = 0; rowIndex < numberOfPerceptrons; ++rowIndex)
		output(rowIndex + 1, 1) = perceptrons[rowIndex].neuron.get();
	return output;
}

void Layer::report()
{
	std::cout << "This layer has " << numberOfPerceptrons << " perceptrons and " << numberOfConnections << " connections from last layer." << std::endl;
}