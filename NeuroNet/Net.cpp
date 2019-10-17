#include "Net.h"
#include <cmath>
#include <algorithm>	//for shuffle the inputs and desired_outputs pair
#include <random>
#include <chrono>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "Neuron.h"
double tanh_prime(double x)
{
	return 1.0 - (tanh(x) * tanh(x));
}


double sigmoid_prime(double x)
{
	return sigmoid(x) * (1.0 - sigmoid(x));
}

Net::Net(size_t numberOfInputs, const std::vector<size_t> numberOfNeuronsEachLayer):numberOfInputs(numberOfInputs),numberOfLayers(numberOfNeuronsEachLayer.size())
{
	layers.push_back(Layer(numberOfNeuronsEachLayer[0], numberOfInputs));
	for (size_t index = 1; index < numberOfNeuronsEachLayer.size(); ++index)
	{
		layers.push_back(Layer(numberOfNeuronsEachLayer[index], layers.back().numberOfPerceptrons));
	}
	local_gradients.resize(numberOfNeuronsEachLayer.size());
}

void Net::set_learning_rate(double rate)
{
	std::cout << "Learning rate set from " << learning_rate << " -> " << rate << std::endl;
	learning_rate = rate;
}

void Net::set_momentum(double momentum)
{
	this->momentum = momentum;
	momentum_flag = true;
}

Matrix Net::init_input(const Matrix& _input) const
{
	if (_input.row()!= numberOfInputs)
	{
		std::cout << "The input matrix of size: ";
		_input.reportSize();
		std::cout << " can't init.\n";
	}
	Matrix temp(_input.row() + 1, 1);
	temp(1, 1) = 1;
	for (size_t rowIndex = 1; rowIndex <= numberOfInputs; ++rowIndex)
	{
		temp(rowIndex + 1, 1) = _input(rowIndex, 1);
	}
	return temp;
}

void Net::init_inputs(const std::vector<Matrix>& _inputs)
{
	for (auto& each_input : _inputs)
		inputs.push_back(init_input(each_input));
}

void Net::init_weight(const std::vector<Matrix> weights, const std::vector<Matrix>& biases)
{
	if (weights.size() != numberOfLayers)
	{
		std::cout << "The vector of weight matrcies size is " << weights.size() << " but there are " << numberOfLayers << " layers to init!\n";
		return;
	}
	if (biases.size() != numberOfLayers)
	{
		std::cout << "The vector of weight matrcies size is " << biases.size() << " but there are " << numberOfLayers << " layers to init!\n";
		return;
	}
	size_t i = 0;
	layers.back().setOutputFlag();
	for (size_t layerIndex=0; layerIndex<layers.size();++layerIndex)
	{
		if (weights[layerIndex].row() != layers[layerIndex].numberOfPerceptrons || weights[layerIndex].column() != layers[layerIndex].numberOfConnections||biases[layerIndex].row()!=layers[layerIndex].numberOfPerceptrons||biases[layerIndex].column()!=1)
		{
			std::cout << "The " << layerIndex << " -th weight matrix to init is of wrong size: ";
			weights[layerIndex].reportSize();
			std::cout << std::endl << "But the layer: "; 
			layers[layerIndex].report();
			return;
		}
		layers[layerIndex].set(weights[layerIndex], biases[layerIndex]);
		weight_k_1.push_back(layers[layerIndex].get_weight());
		local_gradients[layerIndex].resize(weights[layerIndex].row(), 1);
	}
}

void Net::set_desired_outputs(const std::vector<Matrix> _desired_outputs)
{
	desired_outputs = _desired_outputs;
}

Matrix Net::ForwardComputation(size_t k)
{
	if (desired_outputs.size() != inputs.size())
	{
		std::cout << "The batch only contains" << desired_outputs.size() << " desired outputs but there are " << inputs.size() << " inputs!\n";
		abort();
	}
	if (!layers[layers.size() - 1].isOutput())
	{
		std::cout << "The last layer is not set to output mode!\n";
		abort();
	}
	layers[0].run(inputs[k]);
	Matrix out0 = layers[0].output();
	size_t layerIndex = 0;
	for (size_t layerIndex=1;layerIndex<layers.size();++layerIndex)
	{
		if (!layers[layerIndex].isReady())
		{
			std::cout << "Layer" << layerIndex << " in " << k << " iteration is not ready!\n";
			abort();
		}
		else
		{
			layers[layerIndex].run(out0);
			out0 = layers[layerIndex].output();
		}
	}
	return out0;
}

Matrix Net::ForwardComputation(const Matrix& processed_input)
{
	layers[0].run(processed_input);
	Matrix out0 = layers[0].output();
	size_t layerIndex = 0;
	for (size_t layerIndex = 1; layerIndex < layers.size(); ++layerIndex)
	{
		if (!layers[layerIndex].isReady())
		{
			std::cout << "Layer" << layerIndex << " a single forward computation is not ready!\n";
			abort();
		}
		else
		{
			layers[layerIndex].run(out0);
			out0 = layers[layerIndex].output();
		}
	}
	return out0;
}



Matrix Net::getError(size_t k)
{
	using namespace std;
	if (desired_outputs[k].row() != layers[numberOfLayers - 1].numberOfPerceptrons|| desired_outputs[k].column()!=1)
	{
		cout << "The size of desired Output is wrong!\n";
		desired_outputs[k].reportSize();
		cout << "There are " << layers[numberOfLayers - 1].numberOfPerceptrons << " neurons in the output layer!\n";
	}
	Matrix error(desired_outputs[k].row(), 1);
	for (size_t i = 0; i < desired_outputs[k].row(); ++i)
	{
		error(i + 1, 1) = desired_outputs[k](i + 1, 1) - layers[layers.size()-1].output()(i+2,1);
	}
	return error;
}

bool Net::checkResult(const Matrix& output, const Matrix& desired) const
{
	
	return false;
}



void Net::BackwardComputation(size_t k)
{
	Matrix error = getError(k);
	//std::cout <<"Error vector=\n"<< error << std::endl;
	for (size_t i = 0; i < layers.back().numberOfPerceptrons; ++i)
	{
		local_gradients.back()(i + 1, 1) = error(i + 1, 1) * sigmoid_prime(layers.back().get_neuron_origin_sum(i));
	}
	//layer (0-numberofLayers-2)
	for (int layerIndex = numberOfLayers - 2; layerIndex >= 0; --layerIndex)
	{
		// (tanh x)'=1-(tanh x)^2
		// neurons 0-numberOfPerceptrons
		for (size_t rowIndex = 0; rowIndex < layers[layerIndex].numberOfPerceptrons; ++rowIndex)
		{
			double sum = 0;
			//sum up the gradient from back layer
			for (size_t i = 0; i < layers[layerIndex + 1].numberOfPerceptrons; ++i)
				sum += local_gradients[layerIndex + 1](i + 1, 1) * layers[layerIndex + 1].get_weight()(i + 1, rowIndex+2);
			local_gradients[layerIndex](rowIndex + 1, 1) = tanh_prime(layers[layerIndex].get_neuron_origin_sum(rowIndex)) * sum;
		}
	}
	/*We have local_gradients, now compute the updated weights*/
		//for layer 0 - numberofLayer-1
	for (size_t layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
	{
		Matrix updatedWeight;
		if (momentum_flag)
		{
			if (layerIndex == 0)
				updatedWeight = layers[layerIndex].get_weight() * (1 + momentum) - (weight_k_1[layerIndex] * momentum) + ((local_gradients[layerIndex] * inputs[k].transpose()) * learning_rate);
			else
				updatedWeight = layers[layerIndex].get_weight() * (1 + momentum) - (weight_k_1[layerIndex] * momentum) + ((local_gradients[layerIndex] * layers[layerIndex - 1].output().transpose()) * learning_rate);
		}
		else
		{
			if (layerIndex == 0)
				updatedWeight = layers[layerIndex].get_weight()+ ((local_gradients[layerIndex] * inputs[k].transpose()) * learning_rate);
			else
				updatedWeight = layers[layerIndex].get_weight()+ ((local_gradients[layerIndex] * layers[layerIndex - 1].output().transpose()) * learning_rate);
		}
		weight_k_1[layerIndex] = layers[layerIndex].get_weight();
		layers[layerIndex].set(updatedWeight);
		//Matrix updatedWeight(layers[layerIndex].numberOfPerceptrons, layers[layerIndex].numberOfConnections + 1);
		//if (layerIndex == 0)
		//{
		//	for (size_t j = 0; j < updatedWeight.row(); ++j)
		//	{
		//		for (size_t i = 0; i < updatedWeight.column(); ++i)
		//		{
		//			updatedWeight(j + 1, i + 1) = layers[layerIndex].get_weight()(j + 1, i + 1) + learning_rate * local_gradients[layerIndex](j + 1, 1) * inputs[k](i + 1, 1);
		//		}
		//	}
		//}
		//else
		//{
		//	for (size_t j = 0; j < updatedWeight.row(); ++j)
		//	{
		//		for (size_t i = 0; i < updatedWeight.column(); ++i)
		//		{
		//			updatedWeight(j + 1, i + 1) = layers[layerIndex].get_weight()(j + 1, i + 1) + learning_rate * local_gradients[layerIndex](j + 1, 1) * layers[layerIndex-1].output()(i + 1, 1);
		//		}
		//	}
		//}
		//layers[layerIndex].set(updatedWeight);
	}
}

void Net::run(size_t epoch, size_t iterations, bool showFinalWeights, bool showEachPass)
{
	using namespace std;
	//default_random_engine rd;
	//uniform_int_distribution<unsigned> random_index(0, inputs.size() - 1);
	if (inputs.size() != desired_outputs.size())
	{
		std::cout << "Error! There are unequal number of inputs and outputs pairs!\n";
		abort();
	}
	size_t pass = 0;
	for (size_t t = 0; t < epoch; ++t)
	{
		for (size_t i = 0; i < iterations; ++i)
		{
			if (showEachPass)
			{
				cout << "Pass " << pass++ << ":\n";
				Matrix temp = ForwardComputation(i);
				for (size_t j = 1; j < temp.row(); ++j)
				{
					cout << desired_outputs[i](j, 1) << "\t" << temp(j + 1, 1) << endl;
				}
				cin.get();
			}
			else
				ForwardComputation(i);
			BackwardComputation(i);
			//show_local_gradients();
		}
	}
	if (showFinalWeights)
	{
		cout << "final weight:\n\n";
		for (auto& each_layer : layers)
		{
			each_layer.ShowWeight();
			cout << endl;
		}
	}
}


//void Net::update_weights()
//{
//	//for layer 0 - numberofLayer-1
//	for (size_t layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
//	{
//		Matrix updatedWeight(layers[layerIndex].numberOfPerceptrons, layers[layerIndex].numberOfConnections + 1);
//		//for each neuron 
//		for (size_t neuronIndex = 0; neuronIndex < layers[layerIndex].numberOfPerceptrons; ++neuronIndex)
//		{
//			//for each connection to the layer
//			for (size_t connectionIndex = 0; connectionIndex < layers[layerIndex].numberOfConnections+1; ++connectionIndex)
//			{
//				if (connectionIndex == 0)
//					updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * 1;
//				else
//				{
//					if (layerIndex == 0)
//						updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * input(connectionIndex + 1, 1);
//					else
//						updatedWeight(neuronIndex + 1, connectionIndex + 1) = layers[layerIndex].get_weight()(neuronIndex + 1, connectionIndex + 1) + learning_rate * local_gradients[layerIndex](neuronIndex + 1, 1) * layers[layerIndex - 1].get_neuron_output(connectionIndex-1);
//				}
//			}
//		}
//		layers[layerIndex].set(updatedWeight);
//	}
//}


//////////////////////////////////////////Debug//////////////////////////////////////////
void Net::show_all_weights()
{
	for (auto& each_layer : layers)
	{
		each_layer.get_weight().reportSize();
		std::cout << std::endl;
		//each_layer.ShowWeight();
		std::cout << std::endl;
	}
	inputs[0].transpose().reportSize();
	local_gradients[0].reportSize();
	std::cout << std::endl;
	layers[0].output().transpose().reportSize();
}

void Net::show_local_gradients()
{
	using namespace std;
	for (auto& each_local_gradient : local_gradients)
	{
		cout << each_local_gradient << endl;
	}
}

void Net::show_layers()
{
	for (auto& each_layer : layers)
	{
		each_layer.report();
		std::cout << std::endl;
	}
}

void Net::show_input_desired_output_pair()
{
	using namespace std;
	if (inputs.size() != desired_outputs.size())
	{
		cout << "Inputs.size()!=desired_outputs.size()\n";
		return;
	}
	for (size_t index = 0; index < inputs.size(); ++index)
	{
		cout << index << ":\n";
		cout << inputs[index] << endl << desired_outputs[index] << endl;
	}
}

void Net::show_inputs_outputs_size()
{
	std::cout << "Input size = " << inputs.size() << std::endl;
	if (inputs.size() != desired_outputs.size())
		std::cout << "Warning! There are unequal number of inputs and outputs pairs!\n";
}

void Net::show_inputs(size_t index)
{
	////std::cout << inputs[index] << std::endl;

	///*show the testing image in OpenCV*/
	//cv::Mat m2(14, 14, CV_8UC1);
	//int index2 = 2;
	//for (size_t row = 0; row < 14; ++row)
	//{
	//	for (size_t col = 0; col < 14; ++col)
	//	{
	//		m2.at<uchar>(row, col) = inputs[index](index2++, 1)*255.0;
	//	}
	//}
	//cv::namedWindow("Figure");
	//cv::imshow("Figure", m2);
	//cv::waitKey(0);
	//cv::destroyWindow("Figure");
}

void Net::show_desired_outputs(size_t index)
{
	std::cout << desired_outputs[index] << std::endl;
}


/////////////////////////////////////////////////*Test*///////////////////////////////////////////////////////////////
void Net::test(size_t input_index)
{
	using namespace std;
	cout << "\nTarget outputs:\n" << desired_outputs[input_index] << "Neural Net output:\n";
	cout << ForwardComputation(input_index).slice(2,layers[layers.size()-1].numberOfPerceptrons+1,1,1);
}

std::pair<double, bool> Net::test(const Matrix& input, const Matrix& desired_output, unsigned char label_r, bool showEachPass)
{
	using namespace std;
	if(showEachPass)
		cout << "\nTarget outputs:\n" << desired_output << "Neural Net output:\n";
	Matrix result = ForwardComputation(init_input(input)).slice(2, layers[layers.size() - 1].numberOfPerceptrons + 1, 1, 1);
	if (showEachPass)
		cout << result;
	/*calculate error*/
	Matrix error = desired_output - result;
	double sum = 0;
	double max = -1;
	unsigned char predicted = 10;
	for (size_t i = 0; i < 10; ++i)
	{
		sum += abs(error(i + 1, 1));
		if (result(i + 1, 1) > max)
		{
			max = result(i + 1, 1);
			predicted = i;
		}
	}
	bool correct = (predicted == label_r) ? true : false;
	if (showEachPass)
	{
		cout << "\npredicted=" << (int)predicted << "\t real=" << (int)label_r << endl;
		cout << (correct ? "correct" : "wrong!") << endl;
	}
	return std::make_pair(sum, correct);
}

struct metaData 
{

	double learning_rate;
	double momentum;	
	size_t inputs;
	size_t layers_num;
	std::vector<size_t> layers;
};

bool Net::WriteToFile(const std::string fileName)
{
	using namespace std;
	ofstream metaFile(fileName, ios_base::binary);
	ofstream dataFile(fileName + "data", ios_base::binary);
	if (!metaFile.is_open()||!dataFile.is_open())
	{
		metaFile.close();
		dataFile.close();
		return false;
	}
	/*Write Meta data*/
		//metaFile << "Inputs=" << numberOfInputs << endl;
		//metaFile << "Layers=";
		//for (auto& layer : layers)
		//	metaFile << layer.numberOfPerceptrons << ",";
		//metaFile << "\b" << endl;		//?
		//metaFile << "Learning rate=" << learning_rate << endl;
		//metaFile << "Momentum=" << momentum << endl;
	metaData data{ learning_rate, momentum, numberOfInputs };
	for (auto& layer : layers)
		data.layers.push_back(layer.numberOfPerceptrons);
	data.layers_num = layers.size();
	metaFile.write((char*)&data, 2 * (sizeof(double) + sizeof(size_t)));
	metaFile.write((char*)&data.layers, data.layers_num * sizeof(layers.front()));
	/*write data*/
	for (auto& layer : layers)
		dataFile.write((char*)&layer.get_weight(), sizeof(layer.get_weight()));

	/*close file stream*/
	metaFile.close();
	dataFile.close();
	return true;
}

std::unique_ptr<Net> ReadFromFile(const std::string fileName)
{
	using namespace std;
	ifstream metaFile(fileName, ios_base::binary);
	ifstream dataFile(fileName + "data", ios_base::binary);
	if (!metaFile.is_open() || !dataFile.is_open())
	{
		metaFile.close();
		dataFile.close();
		return nullptr;
	}
	metaData data;
	metaFile.read((char*)&data, 2 * (sizeof(double)+sizeof(size_t)));
	metaFile.read((char*)&data.layers, data.layers_num * sizeof(size_t));
	unique_ptr<Net> pt{ make_unique<Net>(data.inputs, data.layers) };
	pt->set_learning_rate(data.learning_rate);
	if (data.momentum != 0)
		pt->set_momentum(data.momentum);
	metaFile.close();
	dataFile.close();
	return pt;
}
