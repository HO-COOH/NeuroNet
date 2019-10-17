#include "Neuron.h"
#include <cmath>

double sigmoid(double sum)
{
	return 1.0 / (1.0 + exp(-sum));
}

void Neuron::set(double sum)
{
	if (outputFlag == false)
		value = tanh(sum);
	else
		value = sigmoid(sum);
	this->sum =sum;
}

double Neuron::get() const
{
	return value;
}

