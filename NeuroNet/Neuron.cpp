#include "Neuron.h"
#include <cmath>
void Neuron::set(double sum)
{
	value = tanh(sum);
}

double Neuron::get()
{
	return value;
}
