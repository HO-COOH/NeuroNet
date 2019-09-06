#include <cmath>
#include <iostream>
#include "Matrix.h"
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
}

int main()
{
	MatrixTest();
}