#include "Neuron.h"
#include <cmath>
void Neuron::set(double sum)
{
	if (outputFlag == false)
		value = tanh(sum);
	else
		value = sum;
}

double Neuron::get()
{
	return value;
}

void OutputNeuron::set(double sum)
{
	value = sum;
}
