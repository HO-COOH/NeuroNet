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
	double get() const;
	double sum = 0;
};

double sigmoid(double sum);
