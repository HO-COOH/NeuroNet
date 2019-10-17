#include <cmath>
#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "Net.h"
#include <iterator>
#include "Timer.h"
#include "Reader.h"
#include <opencv2/opencv.hpp>

using namespace std;

void showMatirx(const Matrix& m)
{
	cv::Mat m2(14, 14, CV_8UC1);
	int index2 = 1;
	for (size_t row = 0; row < 14; ++row)
	{
		for (size_t col = 0; col < 14; ++col)
		{
			m2.at<uchar>(row, col) = m(index2++, 1) * 255.0;
		}
	}
	cv::namedWindow("Figure");
	cv::imshow("Figure", m2);
	cv::waitKey(0);
	cv::destroyWindow("Figure");
}



void MatrixTest()
{
	cout << eye(5).transpose() * 2.1;
}

/*Project 1*/
void NetTest()
{
	Timer t;
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
	test.set_learning_rate(0.1);
	test.init_inputs(inputs);	
	test.init_weight(weights, bias);
	test.set_desired_outputs(desiredOutputs);

	//test.show_layers();
	//test.show_all_weights();

	test.run(1,8, true);

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
	bias.push_back(random(4, 1));
	bias.push_back(random(1, 1));

	Net test(2, { 4,1 });

	test.init_inputs(inputs);
	test.init_weight(weights, bias);	
	//test.show_all_weights();
	test.set_desired_outputs(outputs);

	test.run(4000,4);

	for (size_t i = 0; i < 4; ++i)
		test.test(i);
}

void ReaderTest()
{
	Reader r("train-images.idx3-ubyte", TRANING_IMAGE);
	Reader r2("train-labels.idx1-ubyte", TRAINING_LABEL);
	for (size_t i = 0; i < 60000; ++i)
	{
		Matrix temp(196, 1);
		r >> temp;
		//showMatirx(temp);
		cout << r.inFile.tellg()<<endl;		
		Matrix temp2(10, 1);
		r2 >> temp2;
		cout << temp2;
		cout << r2.inFile.tellg()<<endl;
	}

}

void MNIST()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	Reader train_image("train-images.idx3-ubyte", TRANING_IMAGE);
	Reader train_label("train-labels.idx1-ubyte", TRAINING_LABEL);
	Net net(196, { 100,10 });
	train_image >> net;	//init inputs
	train_label >> net;	//init desired_outputs
	vector<Matrix>weights;	//init weights
	weights.push_back(zeros(100, 196));
	weights.push_back(zeros(10, 100));
	vector<Matrix>bias;
	bias.push_back(random(100, 1));
	bias.push_back(random(10, 1));
	net.init_weight(weights, bias);
	net.show_inputs_outputs_size();

	net.WriteToFile("test");
	{
		double sum = 0;
		Reader test_image("t10k-images.idx3-ubyte", TEST_IMAGE);
		Reader test_label("t10k-labels.idx1-ubyte", TEST_LABEL);
		for (size_t i = 0; i < 10000; i++)	//change i for testing times here
		{
			Matrix input(196, 1);
			Matrix label(10, 1);
			test_image >> input;
			test_label >> label;
			sum += net.test(input, label).first;
		}
		cout << "Before training, error=" << sum << endl;
	}

	size_t epoch = 0;
	do 
	{
		net.run(1, 60000, false, false);
		//cout << "Training finished! Any key to test:";
		//cin.get();

		double sum = 0;	
		Reader test_image("t10k-images.idx3-ubyte", TEST_IMAGE);
		Reader test_label("t10k-labels.idx1-ubyte", TEST_LABEL);
		Reader test_label_value("t10k-labels.idx1-ubyte", TEST_LABEL);
		size_t correct_count = 0;
		for (size_t i = 0; i < 10000; i++)	//change i for testing times here
		{
			Matrix input(196, 1);
			Matrix label(10, 1);
			unsigned char label_r;
			test_label_value >> label_r;
			test_image >> input;
			//showMatirx(input);
			test_label >> label;
			//cout << label;
			auto result=net.test(input, label,label_r,false);
			sum += result.first;
			if (result.second)
				++correct_count;
		}
		double acc = correct_count / 10000.0;
		cout << "Finished " << epoch++ << " epoches. " << "error=" << sum <<"\tacc="<<acc<< endl;
	} while (epoch<10);

	/*Demo*/
	Reader test_image("t10k-images.idx3-ubyte", TEST_IMAGE);
	Reader test_label("t10k-labels.idx1-ubyte", TEST_LABEL);
	Reader test_label_value("t10k-labels.idx1-ubyte", TEST_LABEL);
	for (size_t i = 0; i < 10000; i++)	//change i for testing times here
	{
		Matrix input(196, 1);
		Matrix label(10, 1);
		test_image >> input;
		test_label >> label;
		unsigned char label_r;
		test_label_value >> label_r;
		net.test(input, label, label_r, true);
		showMatirx(input);
	}
}

int main()
{
	//MatrixTest();
	//NetTest();
	//ReaderTest();
	MNIST();
	cout << endl;
	cout << endl;
	cout << "\n\nFinished!\n";
	cin.get();
	//XORTest();
}