/*
Author: Dan Rehberg
Modified Date: 12/9/2022
Purpose: Serial Matrix class build from the prior tested ParallelMatrix class.
	2D matrix for the Neural Network structure being built.
	2D works because it is merely vectors of features, stacked by the number of samples gathered.
*/

#ifndef __SERIAL_MATRIX__
#define __SERIAL_MATRIX__

#include <array>
#include <string>
#include <vector>
#include <utility>
#include <ostream>
#include <mutex>
#include <atomic>
#include <new>
#include <exception>
#include <cmath>

//Is just a 2D matrix class to start playing around with Neural Networks in C++
class SerialMatrix
{
public:
	SerialMatrix();//Used to explicitly instantiate variables
	SerialMatrix(size_t rows, size_t columns);//POD, could used fixed length if so desired
	template <size_t T, size_t S>
	SerialMatrix(const std::array<std::array<float, S>, T>& array2D)
	{
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array copy) Cannot have a matrix with zero elements");
		deepCopy(array2D);
	}
	SerialMatrix(const std::vector<std::vector<float>>& vector2D);
	SerialMatrix(const SerialMatrix& cp);
	SerialMatrix(SerialMatrix&& rhs);
	~SerialMatrix();
	SerialMatrix& operator=(const SerialMatrix& cp);//!!DATA MODIFICATION!! -- managed through deep copy
	SerialMatrix& operator=(SerialMatrix&& rhs);
	template <size_t T, size_t S>
	SerialMatrix& operator=(const std::array<std::array<float, S>, T>& array2D)
	{
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array =) Cannot have a matrix with zero elements");
		deepCopy(array2D, false);
	}
	SerialMatrix& operator=(const std::vector<std::vector<float>>& vector2D);
	friend bool operator==(const SerialMatrix& A, const SerialMatrix& B);
	friend SerialMatrix operator-(const float lhs, const SerialMatrix& rhs);
	friend SerialMatrix operator/(const float lhs, const SerialMatrix& rhs);
	friend SerialMatrix operator*(const float lhs, const SerialMatrix& rhs);
	SerialMatrix operator*(const SerialMatrix& rhs) const;
	SerialMatrix operator-(const SerialMatrix& rhs) const;
	SerialMatrix operator/(const SerialMatrix& rhs) const;
	SerialMatrix operator+(const SerialMatrix& rhs) const;
	SerialMatrix& operator+=(const SerialMatrix& rhs);
	size_t getCapacity() const;
	std::pair<size_t, size_t> getDimensions() const;
	std::string getInfo() const;
	void activateTanH();
	static float mean(const SerialMatrix& ref);//total arithmetic mean
	static SerialMatrix mean(const SerialMatrix& ref, bool row);//row or column arithmetic means
	static SerialMatrix standardDeviations(const SerialMatrix& ref, bool row);//row true means mean of each row
	static SerialMatrix square(const SerialMatrix& ref);
	//static SerialMatrix sqaureroot(const SerialMatrix& ref);
	static SerialMatrix addOnes(const SerialMatrix& ref);
	static SerialMatrix transpose(const SerialMatrix& ref, size_t rowStart = 0);
	static SerialMatrix componentwise(const SerialMatrix& A, const SerialMatrix& B);
	//Parallel operations
	static SerialMatrix* mA, * mB, mC;
	static void setParallelMatrixOps(SerialMatrix& matA, SerialMatrix& matB, bool multiplication = true);
	static void parallelDotProducts(std::mutex& m, unsigned int taskIndex);//Dot product managed per thread
	static SerialMatrix&& moveParallelResult();
private:
	size_t rows, columns;//length of 2D matrix
	size_t capacity;//total cardinality of the 2D matrix
	float* data;//the matrix innards
	void deepCopy(const SerialMatrix& cp, bool allocate = true);//!!DATA MODIFICATION!! -- allocate is true, deletes if not nullptr : false, then checks for like parameters to determine deletion
	template <size_t T, size_t S>
	void deepCopy(const std::array<std::array<float, S>, T>& values, bool allocate = true)//!!DATA MODIFICATION!! -- allocate is true, deletes if not nullptr : false, then checks for like parameters to determine deletion
	{
		bool cleanStart = true;
		if (allocate || data == nullptr) {}
		else
		{
			if (capacity == (values.size() * values[0].size()))
			{
				cleanStart = false;
			}
		}
		rows = values.size();
		columns = values[0].size();
		capacity = rows * columns;
		if (cleanStart)
		{
			delete[] data;
			data = new (std::nothrow)float[capacity];
			if (data == nullptr)throw std::bad_alloc();//"Serial Matrix class, bad allocation on STD::Array DeepCopy"
		}
		for (auto itr = values.begin(); itr != values.end(); ++itr)
		{
			for (auto jtr = itr->begin(); jtr != itr->end(); ++jtr)
			{
				//Row major in flattened indexing
				data[(itr - values.begin()) * columns + (jtr - itr->begin())] = *jtr;
			}
		}
	}
	void deepCopy(const std::vector<std::vector<float>>& values, bool allocate = true);
};

std::ostream& operator<<(std::ostream& out, const SerialMatrix& mat);

#endif // !__SERIAL_MATRIX__