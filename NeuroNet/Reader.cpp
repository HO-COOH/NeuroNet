#include "Reader.h"
#include <intrin.h>
Reader::Reader(const std::string& fileName, mode _mode):_mode(_mode)
{
	using namespace std;
	inFile.open(fileName, ios::binary);
	if (!inFile.is_open())		//file open failed
	{
		cout << "The file: " << fileName << " doesn't exist!\n";
		abort();
	}
	int32_t magic_number;
	inFile.read((char*)&magic_number, 4);
	magic_number = _byteswap_ulong(magic_number);	//The magic number in mnist is in high-endian, so flip the order
	switch (_mode)
	{
	case TRANING_IMAGE:	
	case TEST_IMAGE:
		if (magic_number != 0x803)
		{
			cout << "Error! The file name: " << fileName << " is not image file!\n";
			abort();
		}
		inFile.seekg(16);
		break;
	case TRAINING_LABEL:
	case TEST_LABEL:
		if (magic_number != 0x801)
		{
			cout << "Error! The file name: " << fileName << " is not label file!\n";
			abort();
		}
		inFile.seekg(8);
		break;
	}
}

Reader& Reader::operator>>(Matrix& m)
{
	if (!inFile)
	{
		std::cout << "Error! Reached end of the file!\n";
		return *this;
	}
	if (_mode == TRANING_IMAGE || _mode == TEST_IMAGE)	//reading images and down sample to 14*14 images
	{
		if (m.row() != 196)
			std::cout << "Matrix size error in reading image\n";
		int rowIndex = 1;
		for (char row = 0; row < 28; row+=2 )
		{
			unsigned char data[28];
			inFile.read((char*)data, 28);
			for (char i = 0; i < 28; i+=2)
				m(rowIndex++, 1) = (double)data[i]/255.0;		//if it is image, the itensity of every pixel needs to be normalized (divided by 255)
			inFile.seekg(28, std::ios_base::cur);
		}
	}
	else	//reading labels
	{
		if (m.row() != 10)
			m.resize(10, 1);
		unsigned char value;
		inFile.read((char*)&value, 1);
		m(value + 1, 1) = 1.0;
	}
	return *this;
}


Reader& Reader::operator>>(Net& n)
{
	switch (_mode)
	{
	case TRANING_IMAGE:
	{
		unsigned training_samples = 60000;
		n.inputs.reserve(training_samples);
		for (unsigned i = 0; i < training_samples; ++i)
		{
			Matrix temp(196, 1);
			*this>>temp;
			n.inputs.emplace_back(n.init_input(temp));
		}
		break;
	}
	case TRAINING_LABEL:
	{
		unsigned training_samples = 60000;
		n.desired_outputs.reserve(training_samples);
		for (unsigned i = 0; i < training_samples; ++i)
		{
			Matrix temp(10, 1);
			*this >> temp;
			n.desired_outputs.emplace_back(temp);
		}
		break;
	}
	default:
		std::cout << "Testing labels needs to be manually input!\n";
		break;
	}
	return *this;
}

Reader& Reader::operator>>(unsigned char& label)
{
	switch (_mode)
	{
	case TRANING_IMAGE:
		break;
	case TRAINING_LABEL:
		break;
	case TEST_IMAGE:
		break;
	case TEST_LABEL:
		inFile.read((char*)&label, 1);
		break;
	default:
		break;
	}
	return *this;
}
