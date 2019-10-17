#pragma once
#include <string>
#include "Matrix.h"
#include <fstream>
#include "Net.h"

enum mode { TRANING_IMAGE, TRAINING_LABEL, TEST_IMAGE, TEST_LABEL };
class Reader
{
public:
	std::ifstream inFile;
	mode _mode;
public:
	Reader(const std::string& fileName, mode _mode);
	Reader& operator>>(Matrix& m);
	Reader& operator>>(Net& n);
	Reader& operator>>(unsigned char& label);
	~Reader()
	{
		inFile.close();
	}
};

