#include <cmath>
#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "Net.h"
#include <iterator>

using namespace std;

void MatrixTest()
{
	double arr[] = { 1,2,3,4,5,6,7,8,9 };
	Matrix m1(arr,3,3);
	cout << m1;
	auto m2 = eye(5);
	m2.resize(3,4);
	cout << m2;
	double wei[] = { 0.1,0.2,0.3,0.1,0.1,0.1,0.3,0.3,0.3 };
	Matrix weight1(wei, 3, 3);
	cout << endl << weight1;
	cout << eye(5) / 2;
}

void LayerTest()
{
	Layer a(3,3);
	double wei[] { 0.1,0.2,0.3,0.1,0.1,0.1,0.3,0.3,0.3 };
	Matrix weight(wei, 3, 3);
	double b[]{ 0.2,0.1,0.9 };
	Matrix bias(b, 3, 1);
	a.set(weight,bias);
	double i[]{ 1.0,1.0,1.0 };
	Matrix input(i, 3, 1);
	a.run(input);
	cout << a.output();
	a.report();
}

void NetTest()
{
	double wei1[]{ 0.1,0.2,0.3,0.1,0.1,0.1,0.3,0.3,0.3 };
	double wei2[]{ 0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2 };
	double wei3[]{ 1.5,1.2,1.0,0.0,0.0,0.8,0.1,0.0 };
	double b1[]{ 0.2,0.1,0.9 };
	double b2[]{ 0.0,0.2,0.0,-0.1 };
	double b3[]{ -0.2,-0.1 };
	double in1[]{ 1.0,1.0,1.0 };
	Net test(3, 3, Matrix(in1, 3, 1));
	vector<Matrix> weights(3);
	weights[0]=(Matrix(wei1, 3, 3));
	weights[1] = (Matrix(wei2, 4, 3));
	weights[2] = (Matrix(wei3, 2, 4));
	
	vector<Matrix> bias(3);
	bias[0]=(Matrix(b1, 3, 1));
	bias[1] = (Matrix(b2, 4, 1));
	bias[2] = (Matrix(b3, 2, 1));
	test.init_weight(weights, bias);

	test.run(true);
}

int main()
{
	MatrixTest();
	cout << endl;
	LayerTest();
	cout << endl;
	NetTest();
	cout << "\n\nFinished!\n";
}