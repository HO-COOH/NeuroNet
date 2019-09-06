#pragma once
#include <vector>
#include <iostream>
class Matrix
{
	size_t rows;
	size_t columns;
	std::vector<std::vector<double>>data;
public:
	/*Constructors*/
	Matrix();
	Matrix(size_t _rows, size_t _columns);
	Matrix(double* arr, size_t _rows = 1, size_t _columns = 1);
	Matrix(std::vector<double> rows[], size_t _rows = 1, size_t _columns = 1);
	Matrix(const std::vector<std::vector<double>>& _data);

	/*manipulate*/
	void resize(size_t _rows, size_t _columns);

	/*data access*/
	size_t row() const;
	size_t column() const;
	size_t size() const;
	std::vector<double> rowAt(size_t rowIndex);
	std::vector<double> colAt(size_t colIndex);
	double& operator()(size_t row, size_t col);

	/*Arithmetics*/
	Matrix operator+(const Matrix& m2) const;
	Matrix operator+() const;
	Matrix operator-(const Matrix& m2) const;
	Matrix operator-() const;
	Matrix operator*(const Matrix& m2) const;
	Matrix operator*(double v) const;

	Matrix& operator+=(const Matrix& m2);
	Matrix& operator-=(const Matrix& m2);
	Matrix& operator*=(const Matrix& m2);
	Matrix& operator*=(double v);
	
	friend std::ostream& operator<<(std::ostream& out, const Matrix& m);
};

Matrix eye(size_t size);
Matrix zeros(size_t size);
Matrix zeros(size_t rows, size_t cols);