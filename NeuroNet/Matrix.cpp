#include "Matrix.h"
#include <iterator>

Matrix::Matrix() :rows(0), columns(0)
{}

Matrix::Matrix(size_t _rows, size_t _columns) : rows(_rows), columns(_columns)
{
	data.resize(_rows);
	for (auto& row : data)
	{
		row.resize(_columns);
		for (auto& element : row)
			element = 0;
	}
}

Matrix::Matrix(double* arr, size_t _rows, size_t _columns) :rows(_rows), columns(_columns)
{
	data.resize(_rows);
	for (auto& row : data)
	{
		row.resize(_columns);
	}
	if (arr != nullptr)
	{
		double* current = arr;
		for (auto& row : data)
		{
			std::copy(current, current + _columns, row.begin());
			current += _columns;

		}
	}
}

Matrix::Matrix(std::vector<double> rows[], size_t _rows, size_t _columns) :rows(_rows), columns(_columns)
{
	auto currentRow = rows;
	size_t rowIndex = 0;
	while (rowIndex != _rows && currentRow != nullptr)
	{
		std::copy(currentRow->begin(), currentRow->end(), data[rowIndex].begin());
		++rowIndex;
		++currentRow;
	}
	if (rowIndex != _rows)
		std::cout << "Error in construction in new Matrix in " << rowIndex << " ¡Ù" << _rows;
}

Matrix::Matrix(const std::vector<std::vector<double>>& _data):rows(_data.size()),columns(_data[0].size())
{
	std::copy(_data.begin(), _data.end(), data.begin());
}

void Matrix::resize(size_t _rows, size_t _columns)
{
	Matrix temp(_rows, _columns);
	size_t rowLim = (_rows > rows ? rows : _rows);
	size_t colIndex = 0;
	size_t colLim = (_columns > columns ? columns : _columns);
	for (size_t rowIndex = 0; rowIndex < rowLim; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex < colLim; ++colIndex)
			temp.data[rowIndex][colIndex] = data[rowIndex][colIndex];
	}
	(*this) = temp;
}

inline size_t Matrix::row() const
{
	return rows;
}

inline size_t Matrix::column() const
{
	return columns;
}

inline size_t Matrix::size() const
{
	return rows * columns;
}

std::vector<double> Matrix::rowAt(size_t rowIndex)
{
	return std::vector<double>(data[rowIndex]);
}

std::vector<double> Matrix::colAt(size_t colIndex)
{
	std::vector<double>temp(columns);
	for (auto& row : data)
		temp.push_back(row[0]);
	return temp;
}

inline double& Matrix::operator()(size_t row, size_t col)
{
	return data[row - 1][col - 1];
}

Matrix Matrix::operator+(const Matrix& m2) const
{
	if (rows != m2.rows || columns != m2.columns)
	{
		std::cout << "The matrix of size " << rows << "*" << columns << " can't be added to a "<<m2.rows<<"*"<<m2.columns<<" matrix\n";
		return Matrix();
	}
	Matrix m(*this);
	size_t rowIndex = 0;
	for (auto& row: m.data)
	{
		size_t colIndex = 0;
		for (auto& element : row)
		{
			element += m2.data[rowIndex][colIndex++];
		}
		++rowIndex;
	}
	return m;
}

Matrix Matrix::operator+() const
{
	return Matrix(*this);
}

Matrix Matrix::operator-(const Matrix& m2) const
{
	if (rows != m2.rows || columns != m2.columns)
	{
		std::cout << "The matrix of size " << rows << "*" << columns << " can't be subtracted!\n";
		return Matrix();
	}
	Matrix m(*this);
	for (size_t rowIndex = 0; rowIndex != m2.rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex != m2.columns; ++colIndex)
			m.data[rowIndex][colIndex] -= m2.data[rowIndex][colIndex];
	}
	return m;
}

Matrix Matrix::operator-() const
{
	Matrix m(*this);
	for (size_t rowIndex = 0; rowIndex != m.rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex != m.columns; ++colIndex)
			m.data[rowIndex][colIndex] = -(m.data[rowIndex][colIndex]);
	}
	return m;
}

Matrix Matrix::operator*(const Matrix& m2) const
{
	if (columns != m2.rows)
	{
		std::cout << "Two matrix of size: " << rows << "*" << columns << " and " << m2.rows << "*" << m2.columns << " can't be multiplied!\n";
		return Matrix();
	}
	Matrix m(rows, m2.columns);
	for (size_t rowIndex = 0; rowIndex != m.rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex != m.columns; ++colIndex)
		{
			double& sum = m.data[rowIndex][colIndex] = 0;
			for (size_t i = 0; i < columns; ++i)
			{
				sum += data[rowIndex][i] * m2.data[i][colIndex];
			}
		}
	}
	return m;
}

Matrix Matrix::operator*(double v) const
{
	Matrix m(rows, columns);
	for (size_t rowIndex = 0; rowIndex < rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex < columns; ++colIndex)
		{
			m.data[rowIndex][colIndex] = data[rowIndex][colIndex]*v;
		}
	}
	return m;
}

Matrix& Matrix::operator+=(const Matrix& m2)
{
	for (size_t rowIndex = 0; rowIndex < rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex < columns; ++colIndex)
		{
			data[rowIndex][colIndex] += m2.data[rowIndex][colIndex];
		}
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m2)
{
	for (size_t rowIndex = 0; rowIndex < rows; ++rowIndex)
	{
		for (size_t colIndex = 0; colIndex < columns; ++colIndex)
		{
			data[rowIndex][colIndex] -= m2.data[rowIndex][colIndex];
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m2)
{
	(*this) = (*this) * m2;
	return *this;
}

Matrix& Matrix::operator*=(double v)
{
	(*this) = (*this) * v;
	return *this;
}

std::ostream& operator<<(std::ostream& out, const Matrix& m)
{
	for (auto& row : m.data)
	{
		out << "[";
		std::copy(row.begin(), row.end(), std::ostream_iterator<double>(out, ","));
		out <<'\b' << "]" << std::endl;
	}
	return out;
}

Matrix eye(size_t size)
{
	Matrix temp(size, size);
	for (size_t i = 1; i <= size; ++i)
		temp(i, i) = 1;
	return temp;
}

Matrix zeros(size_t size)
{
	return Matrix(size, size);
}

Matrix zeros(size_t rows, size_t cols)
{
	return Matrix(rows, cols);
}