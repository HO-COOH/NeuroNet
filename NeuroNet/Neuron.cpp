#include "Neuron.h"
#include <cmath>
void Neuron::set(double sum)
{
	if (outputFlag == false)
		value = tanh(sum);
	else
		value = sum;
	this->sum = sum;
}

double Neuron::get() const
{
	return value;
}

