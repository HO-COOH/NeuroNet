#include "Neuron.h"
#include <cmath>
inline void Neuron::set(double sum)
{
	value = tanh(sum);
}

inline double Neuron::get()
{
	return value;
}
