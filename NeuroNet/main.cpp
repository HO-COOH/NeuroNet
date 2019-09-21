#include <cmath>
#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "Net.h"
#include <iterator>
#include "Timer.h"

using namespace std;

/*Project 1*/
void NetTest()
{
	/* 3 weight*/
	double wei1[]{ 0.1,0.2,0.3,0.1,0.1,0.1,0.3,0.3,0.3 };
	double wei2[]{ 0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2 };
	double wei3[]{ 1.5,1.2,1.0,0.0,0.0,0.8,0.1,0.0 };	
	vector<Matrix> weights(3);
	weights[0]=(Matrix(wei1, 3, 3));
	weights[1] = (Matrix(wei2, 4, 3));
	weights[2] = (Matrix(wei3, 2, 4));	
	/* 3 bias*/
	double b1[]{ 0.2,0.1,0.9 };
	double b2[]{ 0.0,0.2,0.0,-0.1 };
	double b3[]{ -0.2,-0.1 };	
	vector<Matrix> bias(3);
	bias[0] = (Matrix(b1, 3, 1));
	bias[1] = (Matrix(b2, 4, 1));
	bias[2] = (Matrix(b3, 2, 1));
	/* 8 input */
	double in1[]{ 0.0,0.0,0.0 };
	double in2[]{ 0.0,0.0,1.0 };
	double in3[]{ 0.0,1.0,0.0 };
	double in4[]{ 0.0,1.0,1.0 };
	double in5[]{ 1.0,0.0,0.0 };
	double in6[]{ 1.0,0.0,1.0 };
	double in7[]{ 1.0,1.0,0.0 };
	double in8[]{ 1.0,1.0,1.0 };
	vector<Matrix> inputs;
	inputs.reserve(8);
	inputs.emplace_back(Matrix(in1, 3, 1));
	inputs.emplace_back(Matrix(in2, 3, 1));
	inputs.emplace_back(Matrix(in3, 3, 1));
	inputs.emplace_back(Matrix(in4, 3, 1));
	inputs.emplace_back(Matrix(in5, 3, 1));
	inputs.emplace_back(Matrix(in6, 3, 1));
	inputs.emplace_back(Matrix(in7, 3, 1));
	inputs.emplace_back(Matrix(in8, 3, 1));
	/*8  desired outputs*/
	double o1[]{1,0};
	double o2[]{ 0,1 };
	double o3[]{ 0,1 };
	double o4[]{ 1,0 };
	double o5[]{ 0,1 };
	double o6[]{ 1,0 };
	double o7[]{ 1,0 };
	double o8[]{ 1,0 };
	vector<Matrix> desiredOutputs;
	desiredOutputs.reserve(8);
	desiredOutputs.emplace_back(Matrix(o1, 2, 1));
	desiredOutputs.emplace_back(Matrix(o2, 2, 1));
	desiredOutputs.emplace_back(Matrix(o3, 2, 1));
	desiredOutputs.emplace_back(Matrix(o4, 2, 1));
	desiredOutputs.emplace_back(Matrix(o5, 2, 1));
	desiredOutputs.emplace_back(Matrix(o6, 2, 1));
	desiredOutputs.emplace_back(Matrix(o7, 2, 1));
	desiredOutputs.emplace_back(Matrix(o8, 2, 1));


	Net test(3, {3,4,2});
	test.init_inputs(inputs);	
	test.init_weight(weights, bias);
	test.set_desired_outputs(desiredOutputs);

	//test.show_layers();
	//test.show_all_weights();

	test.run(10000);

	cout << "\n\n";
	//test.show_input_desired_output_pair();
	
	for (size_t i = 0; i < 8; ++i)
		test.test(i);

}

/*A neural network simulating an XOR gate*/
void XORTest()
{
	/*desired inputs*/
	double in1[]{ 0,0 };
	double in2[]{ 0,1 };
	double in3[]{ 1,0 };
	double in4[]{ 1,1 };
	vector<Matrix> inputs;
	inputs.push_back(Matrix(in1, 2, 1));
	inputs.push_back(Matrix(in2, 2, 1));
	inputs.push_back(Matrix(in3, 2, 1));
	inputs.push_back(Matrix(in4, 2, 1));
	/*desired outputs*/
	double o1[]{ 0 };
	double o2[]{ 1 };
	double o3[]{ 1 };
	double o4[]{ 0 };
	vector<Matrix> outputs;
	outputs.push_back(Matrix(o1, 1, 1));
	outputs.push_back(Matrix(o2, 1, 1));
	outputs.push_back(Matrix(o3, 1, 1));
	outputs.push_back(Matrix(o4, 1, 1));

	/*weights*/
	vector<Matrix>weights;
	weights.push_back(random(4,2));
	weights.push_back(random(1,4));

	/*bias*/
	vector<Matrix> bias;
	bias.push_back(zeros(4, 1));
	bias.push_back(zeros(1, 1));

	Net test(2, { 4,1 });

	test.init_inputs(inputs);
	test.init_weight(weights, bias);	
	//test.show_all_weights();
	test.set_desired_outputs(outputs);

	test.run(1000);

	for (size_t i = 0; i < 4; ++i)
		test.test(i);
}

int main()
{
	Timer t;
	NetTest();
	cout << endl;
	cout << endl;
	cout << "\n\nFinished!\n";
	//XORTest();
}