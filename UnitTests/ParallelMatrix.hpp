/*
Author: Dan Rehberg
Date: 12/9/2022
Purpose: This class is to enable general vector operations in parallel.
	Performance tests will ensue in order to determine if contemporary
		parallel matrix mulitplication can be outdone via atomic operations.

Goals: 4 scenarios exist to run within this class to test out performance.
	- Serial Matrix Multiplications
	- Parallel Stages following the basic log_base2(N) stages
		~ Note, for matrix multiplication this will mean running multiple concurrent dot products in a stage.
	- ~~Parallel with a single Stage, using atomic bools to keep track of which thread is allowed to produce a summation.~~ (Cancelled)
	- Parallel per dot product in the matrix multiplication. (Replacing above)
	- Parallel with atomic integers (2 stages): stage 1 converts floats to atomic int, stage 2 is letting everything do as it pleases.
*/

#ifndef __PARALLEL_MATRIX__
#define __PARALLEL_MATRIX__


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

//Pretty bleak interface below..
//	As appealing as templating might be, would require a virtual function to be overloaded to perform the atomic operations
//	Additionally, do not have any immediate reason to require use of other numeric types
class AtomicMatrix
{
private:
	AtomicMatrix();
public:
	AtomicMatrix(unsigned int rows, unsigned int columns); //Not directly accessible
	AtomicMatrix(AtomicMatrix&& rhs);
	~AtomicMatrix();
	AtomicMatrix& operator=(AtomicMatrix&& rhs);
	friend class ParallelMatrix;
	friend class IntMatrix;

private:
	static constexpr float scale = 10000.0f;
	static constexpr float invScale = 1.0f / float(scale);
	std::atomic<int>* data = nullptr;
};

class IntMatrix
{
private:
	IntMatrix();
public:
	IntMatrix(unsigned int rows, unsigned int columns);
	IntMatrix(IntMatrix&& rhs);
	~IntMatrix();
	IntMatrix& operator=(IntMatrix&& rhs);
	IntMatrix& operator=(const ParallelMatrix& ref);
	friend class ParallelMatrix;
private:
	int* data = nullptr;
};

//Parent Class, if Polymorphism used
//Is just a 2D matrix class to start playing around with Neural Networks in C++
class ParallelMatrix
{
private:
	ParallelMatrix();//Used to explicitly instantiate variables
public:
	ParallelMatrix(unsigned int rows, unsigned int columns);//POD, could used fixed length if so desired
	template <size_t T, size_t S>
	ParallelMatrix(const std::array<std::array<float, S>, T>& array2D)
	{
#if DEBUG_MATRIX >= 1
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array copy) Cannot have a matrix with zero elements");
#endif
		deepCopy(array2D);
	}
	ParallelMatrix(const std::vector<std::vector<float>>& vector2D);
	ParallelMatrix(const ParallelMatrix& cp);//Copy only, no move semantics
	~ParallelMatrix();
	ParallelMatrix& operator=(const ParallelMatrix& cp);//!!DATA MODIFICATION!! -- managed through deep copy
	template <size_t T, size_t S>
	ParallelMatrix& operator=(const std::array<std::array<float, S>, T>& array2D)
	{
#if DEBUG_MATRIX >= 1
		if ((array2D.size() * array2D[0].size()) == 0)throw std::length_error("(array =) Cannot have a matrix with zero elements");
#endif
		deepCopy(array2D, false);
	}
	ParallelMatrix& operator=(const std::vector<std::vector<float>>& vector2D);
	friend bool operator==(const ParallelMatrix& A, const ParallelMatrix& B);
	ParallelMatrix operator*(const ParallelMatrix& rhs) const;
	unsigned int getCapacity() const;
	std::pair<unsigned int, unsigned int> getDimensions() const;
	std::string getInfo() const;
	static ParallelMatrix getParallelResult();
	static void setParallelMatrixOps(ParallelMatrix& matA);
	//TODO -- isolate the parallel operations after testing them via MACRO settings
	static void setParallelMatrixOps(ParallelMatrix& matA, ParallelMatrix& matB, bool multiplication = true);
	static void parallelTrialA(std::mutex& m, unsigned int taskIndex);//Dot product managed per thread
	static void parallelTrialB0(std::mutex& m, unsigned int taskIndex);//Arithmetic operation paired per thread (log base 2 operations)
	static void setTrialB1(unsigned int currentN, bool reset = false);//Changes the curN and alternates the curItr value
	static void parallelTrialB1(std::mutex& m, unsigned int taskIndex);//Addition for log base 2 case
	static void parallelTrialB2(std::mutex& m, unsigned int taskIndex);//Store result for log base 2 case
	static void parallelTrialC0(std::mutex& m, unsigned int taskIndex);//Integer conversion and atomic operations
	static void parallelTrialC0Alt(std::mutex& m, unsigned int taskIndex);
	static void parallelTrialC0AltAlt(std::mutex& m, unsigned int taskIndex);
	static void parallelTrialC1(std::mutex& m, unsigned int taskIndex);//Convert to the float component from the atomic integer value
	friend class IntMatrix;
private:
	static ParallelMatrix* mA, * mB, mC, mD;//C is a resultant, so no pointers; mD is a temporary for the log base 2 dispatch
	static unsigned int M, N, rI, cJ, curN, curItr;
	static AtomicMatrix atom;//for the atomic multiplication tests
	static IntMatrix intA, intB;
#if DEBUG_MATRIX == 2
public:	static std::atomic_uint32_t multiplications;
	  static ParallelMatrix getMD();
private:
#endif
	unsigned int rows, columns;//length of 2D matrix
	unsigned int capacity;//total cardinality of the 2D matrix
	float* data;//the matrix innards
	void deepCopy(const ParallelMatrix& cp, bool allocate = true);//!!DATA MODIFICATION!! -- allocate is true, deletes if not nullptr : false, then checks for like parameters to determine deletion
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
			data = new (std::nothrow)float[capacity];
#if DEBUG_MATRIX >= 1
			if (data == nullptr)std::cout << "PMAT DEEPCOPY ARRAY ALLOCATION FAILURE!\n";
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
				data[(itr - values.begin()) * columns + (jtr - itr->begin())] = *jtr;
			}
		}
	}
	void deepCopy(const std::vector<std::vector<float>>& values, bool allocate = true);
};

//Avoiding a friendship to those that cannot be trusted.. though, I typically trust ostream..
std::ostream& operator<<(std::ostream& out, const ParallelMatrix& mat);

//Could cast to common base class if polymorphism desired 
//Likely just for fixed usage (i.e., 4x4 homogeneous coordinates for graphics and physics of 3D space, etc..)
template <size_t T, size_t S>
class BaseMatrix
{
public:
	BaseMatrix();
	virtual ~BaseMatrix();
	template <size_t V>
	BaseMatrix<T, V> operator*(BaseMatrix<S, V>& rhs);
private:
	float columns[S];
	float rows[T];

};


#endif // !__PARALLEL_MATRIX__