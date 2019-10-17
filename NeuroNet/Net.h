#pragma once
#include <vector>
#include "Layer.h"
#include <utility>
#include <memory>

class Net
{
	std::vector<Layer> layers;
	std::vector<Matrix> local_gradients;
	
	size_t numberOfInputs;
	size_t numberOfLayers;

	std::vector<Matrix> weight_k_1;	//for momentum

	friend class Reader;
	std::vector<Matrix> inputs;
	std::vector<Matrix> desired_outputs;
	
	double learning_rate = 0.01;
	double momentum = 0;
	bool momentum_flag = false;
public:
	Net(size_t numberOfInputs, const std::vector<size_t> numberOfNeuronsEachLayer);
	void set_learning_rate(double rate);
	void set_momentum(double momentum);

	Matrix init_input(const Matrix& _input) const;
	void init_inputs(const std::vector<Matrix>& _inputs);
	void init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases);
	void set_desired_outputs(const std::vector<Matrix> _desired_outputs);

	Matrix ForwardComputation(size_t k);
	Matrix ForwardComputation(const Matrix& processed_input);
	
	Matrix getError(size_t k);
	bool checkResult(const Matrix& output, const Matrix& desired) const;
	void BackwardComputation(size_t k);


	void run(size_t epoch, size_t iterations, bool showFinalWeights=true, bool showEachPass=false);
	
	/*Read and write all the parameters to a file*/
	bool WriteToFile(const std::string fileName);
	friend std::unique_ptr<Net> ReadFromFile(const std::string fileName);

	/*Debug*/
	void show_all_weights();
	void show_local_gradients();
	void show_layers();
	void show_input_desired_output_pair();
	void show_inputs_outputs_size();
	void show_inputs(size_t index);
	void show_desired_outputs(size_t index);
	//void update_weights();
	const std::vector<Matrix>& get_inputs() const { return inputs; }
	const std::vector<Matrix>& get_desired_outputs() const { return desired_outputs; }
	
	//Matrix output;

	/*Test*/
	void test(size_t input_index);
	std::pair<double,bool> test(const Matrix& input, const Matrix& desired_output, unsigned char label_r=10,bool showEachPass=false);
};
