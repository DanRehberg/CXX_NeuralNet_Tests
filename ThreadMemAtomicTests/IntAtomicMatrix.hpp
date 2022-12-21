/*
Author: Dan Rehberg
Date: 12/20/2022
*/

#ifndef __INT_ATOMIC_MATRIX__
#define __INT_ATOMIC_MATRIX__


#define DEBUG_MATRIX 3

#if DEBUG_MATRIX == 1
#include <iostream>
#include <stdexcept>
#endif
#if DEBUG_MATRIX >= 2
#include <iostream>
#endif

#include <array>
#include <string>
#include <vector>
#include <utility>
#include <ostream>
#include <mutex>
#include <atomic>
#include <new>
#include "AtomicMatrix.hpp"

//Parent Class, if Polymorphism used
//Is just a 2D matrix class to start playing around with Neural Networks in C++
class IntegerMatrix
{
private:
	IntegerMatrix();//Used to explicitly instantiate variables
public:
	IntegerMatrix(unsigned int rows, unsigned int columns);//POD, could used fixed length if so desired
	template <size_t T, size_t S>
	IntegerMatrix(const std::array<std::array<float, S>, T>& array2D)
	{
#if DEBUG_MATRIX >= 1
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array copy) Cannot have a matrix with zero elements");
#endif
		deepCopy(array2D);
	}
	IntegerMatrix(const std::vector<std::vector<float>>& vector2D);
	IntegerMatrix(const IntegerMatrix& cp);//Copy only, no move semantics
	~IntegerMatrix();
	IntegerMatrix& operator=(const IntegerMatrix& cp);//!!DATA MODIFICATION!! -- managed through deep copy
	template <size_t T, size_t S>
	IntegerMatrix& operator=(const std::array<std::array<float, S>, T>& array2D)
	{
#if DEBUG_MATRIX >= 1
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array =) Cannot have a matrix with zero elements");
#endif
		deepCopy(array2D, false);
	}
	IntegerMatrix& operator=(const std::vector<std::vector<float>>& vector2D);
	friend bool operator==(const IntegerMatrix& A, const IntegerMatrix& B);
	IntegerMatrix operator*(const IntegerMatrix& rhs) const;
	unsigned int getCapacity() const;
	std::pair<unsigned int, unsigned int> getDimensions() const;
	std::string getInfo() const;
	static IntegerMatrix getParallelResult();
	static void setIntegerMatrixOps(IntegerMatrix& matA);
	//TODO -- isolate the parallel operations after testing them via MACRO settings
	static void setParallelMatrixOps(IntegerMatrix& matA, IntegerMatrix& matB, bool multiplication = true);
	//static void parallelTrialC0AltAlt(MemoryTest& mem, std::mutex& m, unsigned int taskIndex);
	static void parallelTrialC0AltAlt(std::mutex& m, unsigned int startTask, unsigned int endTask);
private:
	static IntegerMatrix* mA, * mB, mC, mD;//C is a resultant, so no pointers; mD is a temporary for the log base 2 dispatch
	static unsigned int M, N, rI, cJ, curN, curItr;
	static AtomicMatrix atom;//for the atomic multiplication tests
	unsigned int rows, columns;//length of 2D matrix
	unsigned int capacity;//total cardinality of the 2D matrix
	int_fast64_t* data;//the matrix innards
	void deepCopy(const IntegerMatrix& cp, bool allocate = true);//!!DATA MODIFICATION!! -- allocate is true, deletes if not nullptr : false, then checks for like parameters to determine deletion
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
#if DEBUG_MATRIX == 1
			std::cout << "DEBUG (deepCopy array): Deleting old Matrix Data\n";
#endif
			delete[] data;
			data = new (std::nothrow)int_fast64_t[capacity];
#if DEBUG_MATRIX >= 1
			if (data == nullptr)throw std::exception("PMAT DEEPCOPY ARRAY ALLOCATION FAILURE!");
#endif
		}
#if DEBUG_MATRIX == 1
		else
			std::cout << "DEBUG (deepCopy array): Repurposing Matrix Data\n";
#endif
		for (auto itr = values.begin(); itr != values.end(); ++itr)
		{
			for (auto jtr = itr->begin(); jtr != itr->end(); ++jtr)
			{
				//Row major in flattened indexing
				data[(itr - values.begin()) * columns + (jtr - itr->begin())] =
					static_cast<int_fast64_t>(AtomicMatrix::scale * (*jtr));
			}
		}
	}
	void deepCopy(const std::vector<std::vector<float>>& values, bool allocate = true);
};

//Avoiding a friendship to those that cannot be trusted.. though, I typically trust ostream..
std::ostream& operator<<(std::ostream& out, const IntegerMatrix& mat);

#endif // !__ATOMIC_MATRIX__