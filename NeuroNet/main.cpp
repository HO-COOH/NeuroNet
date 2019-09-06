#include <cmath>
#include <iostream>
#include "Matrix.h"
#include "Layer.h"
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
}

void LayerTest()
{
	Layer a(3, 3);	
}

int main()
{
	MatrixTest();
	LayerTest();
}