#pragma once
#include <vector>
#include "Layer.h"

class Net
{
	std::vector<Layer> layers;
	std::vector<Matrix> local_gradients;
	
	size_t numberOfInputs;
	size_t numberOfLayers;

	std::vector<Matrix> inputs;
	std::vector<Matrix> desired_outputs;
	
	double learning_rate = 0.1;
public:
	Net(size_t numberOfInputs, const std::vector<unsigned int> numberOfNeuronsEachLayer);

	Matrix init_input(const Matrix& _input);
	void init_inputs(const std::vector<Matrix>& _inputs);
	void init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases);
	void set_desired_outputs(const std::vector<Matrix> _desired_outputs);

	Matrix ForwardComputation(size_t k);
	
	Matrix getError(size_t k);
	void BackwardComputation(size_t k);


	void run(size_t iterations, bool showEachPass=false);

	

	/*Debug*/
	void show_all_weights();
	void show_local_gradients();
	void show_layers();
	void show_input_desired_output_pair();
	//void update_weights();
	
	

	//Matrix output;

	/*Test*/
	void test(size_t input_index);
};

