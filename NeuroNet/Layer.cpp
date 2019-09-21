#include "Layer.h"
#include "Matrix.h"
#include <iostream>

Layer::Layer(size_t _numberOfPerceptrons, size_t _numberOfConnections):perceptrons(_numberOfPerceptrons), numberOfConnections(_numberOfConnections), numberOfPerceptrons(_numberOfPerceptrons), weight(_numberOfPerceptrons,_numberOfConnections+1)
{
}


Layer::Layer(const Matrix& _weight):weight(_weight), numberOfConnections(_weight.column()-1), numberOfPerceptrons(_weight.row())
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
	Matrix newWeight(numberOfPerceptrons, numberOfConnections + 1);
	//set the first col -> bias
	for (size_t rowIndex = 0; rowIndex < numberOfPerceptrons; ++rowIndex)
	{
		newWeight(rowIndex+1, 1) = bias(rowIndex+1, 1);
		for (size_t colIndex = 1; colIndex <= numberOfConnections; ++colIndex)
		{
			newWeight(rowIndex + 1, colIndex + 1) = weight(rowIndex + 1, colIndex);
		}
	}
	this->weight = newWeight;
}

void Layer::set(const Matrix& weight)
{
	using namespace std;
	if (weight.row() != numberOfPerceptrons || weight.column() != numberOfConnections + 1)
	{
		cout << "The weight set to this layer is of wrong size!\n";
		weight.reportSize();
		return;
	}
	this->weight = weight;
}

void Layer::run(const Matrix& lastLayerOutput)
{

	if (lastLayerOutput.row() != numberOfConnections+1 || lastLayerOutput.column() != 1)
	{
		std::cout << "The size of last layer output matrix ";
		lastLayerOutput.reportSize();
		std::cout << " is wrong!\n";
		return;
	}
	//add 1 -> [1, lastLayerOutput]^T

	Matrix result = weight * lastLayerOutput;
	if (result.row() != numberOfPerceptrons || result.column() != 1)
	{
		std::cout << "Error happens in layer.run()!\n";
		return;
	}
	for (size_t rowIndex = 0; rowIndex < numberOfPerceptrons; ++rowIndex)
	{
		if (outputFlag == true)
			perceptrons[rowIndex].setOutputFlag();
		perceptrons[rowIndex].set(result(rowIndex + 1, 1));
	}
}

Matrix Layer::output() const
{
	Matrix output(numberOfPerceptrons+1, 1);
	output(1, 1) = 1;
	for (size_t rowIndex = 1; rowIndex < numberOfPerceptrons+1; ++rowIndex)
		output(rowIndex + 1, 1) = perceptrons[rowIndex-1].get();
	return output;
}

void Layer::report() const
{
	std::cout << "This layer has " << numberOfPerceptrons << " perceptrons and " << numberOfConnections << " connections per neuron from last layer." << std::endl;
}

bool Layer::isReady() const
{
	if(weight.column() != numberOfConnections+1 || weight.row() != numberOfPerceptrons ||perceptrons.size()==0||perceptrons.empty())
		return false;
	return true;
}
