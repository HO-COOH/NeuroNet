#pragma once
class Neuron
{
protected:
	double value = 0;
	bool outputFlag = false;
public:
	void setOutputFlag()
	{
		outputFlag = true;
	}
	void set(double sum);
	double get();
};

class OutputNeuron: public Neuron
{
public:
	void set(double sum);
};