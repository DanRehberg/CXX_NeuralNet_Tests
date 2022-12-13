/*
Author: Dan Rehberg
Modified Date: 12/9/2022
*/
#include "ParallelMatrix.hpp"

//static data values
ParallelMatrix* ParallelMatrix::mA = nullptr;
ParallelMatrix* ParallelMatrix::mB = nullptr;
ParallelMatrix ParallelMatrix::mC = ParallelMatrix(1, 1);
ParallelMatrix ParallelMatrix::mD = ParallelMatrix(1, 1);
unsigned int ParallelMatrix::M = 0;
unsigned int ParallelMatrix::N = 0;
unsigned int ParallelMatrix::rI = 0;
unsigned int ParallelMatrix::cJ = 0;
unsigned int ParallelMatrix::curN = 0;
unsigned int ParallelMatrix::curItr = 0;
AtomicMatrix ParallelMatrix::atom = std::move(AtomicMatrix(1, 1));
IntMatrix ParallelMatrix::intA = std::move(IntMatrix(1, 1));
IntMatrix ParallelMatrix::intB = std::move(IntMatrix(1, 1));

#if DEBUG_MATRIX == 2
std::atomic_uint32_t ParallelMatrix::multiplications = 0;
#endif

AtomicMatrix::AtomicMatrix() : data(nullptr) {}

AtomicMatrix::AtomicMatrix(unsigned int rows, unsigned int columns) : AtomicMatrix()
{
	unsigned int capacity = rows * columns;
#if DEBUG_MATRIX >= 1
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
#endif
	data = new (std::nothrow)std::atomic<int>[capacity];
#if DEBUG_MATRIX >= 1
	if (data == nullptr)std::cout << "ATOMIC ALLOCATION FAILURE!\n";
#endif
	if (data == nullptr)return;
	for (unsigned int i = 0; i < capacity; ++i)
	{
		data[i].store(0, std::memory_order_relaxed);
	}
}

AtomicMatrix::AtomicMatrix(AtomicMatrix&& rhs)
{
	this->data = rhs.data;
	rhs.data = nullptr;
}

AtomicMatrix::~AtomicMatrix()
{
	if (this->data != nullptr)
	{
		delete[] this->data;
	}
	this->data = nullptr;
}

AtomicMatrix& AtomicMatrix::operator=(AtomicMatrix&& rhs)
{
	if (data != nullptr)delete[] data;
	this->data = rhs.data;
	rhs.data = nullptr;
	return *this;
}

IntMatrix::IntMatrix() : data(nullptr) {}

IntMatrix::IntMatrix(unsigned int rows, unsigned int columns) : IntMatrix()
{
	unsigned int capacity = rows * columns;
#if DEBUG_MATRIX >= 1
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
#endif
	data = new (std::nothrow)int[capacity];
#if DEBUG_MATRIX >= 1
	if (data == nullptr)std::cout << "INT ALLOCATION FAILURE!\n";
#endif
}

IntMatrix::IntMatrix(IntMatrix&& rhs)
{
	this->data = rhs.data;
	rhs.data = nullptr;
}

IntMatrix::~IntMatrix()
{
	if (this->data != nullptr)
	{
		delete[] this->data;
	}
	this->data = nullptr;
}

IntMatrix& IntMatrix::operator=(IntMatrix&& rhs)
{
	this->data = rhs.data;
	rhs.data = nullptr;
	return *this;
}

IntMatrix& IntMatrix::operator=(const ParallelMatrix& ref)
{
	//Just going ahead and reallocating this object as needed
	if (this->data != nullptr)delete[] this->data;
	this->data = new (std::nothrow)int[ref.capacity];
	if (this->data == nullptr)*this;
	for (unsigned int i = 0; i < ref.capacity; ++i)
		this->data[i] = static_cast<int>(AtomicMatrix::scale * ref.data[i]);
	return *this;
}

ParallelMatrix::ParallelMatrix() : rows(0), columns(0), capacity(0), data(nullptr)
{

}

ParallelMatrix::ParallelMatrix(unsigned int r, unsigned int c) : ParallelMatrix()
{
	rows = r;
	columns = c;
	capacity = r * c;
#if DEBUG_MATRIX >= 1
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
#endif
	data = new (std::nothrow)float[capacity];
#if DEBUG_MATRIX >= 1
	if (data == nullptr)std::cout << "PMAT CONSTRUCTOR ALLOCATION FAILURE!\n";
#endif
}

ParallelMatrix::ParallelMatrix(const std::vector<std::vector<float>>& values) : ParallelMatrix()
{
#if DEBUG_MATRIX >= 1
	if ((values.size() * values[0].size()) == 0)throw std::length_error("(vector copy) Cannot have a matrix with zero elements");
	size_t consistent = values[0].size();
	for (auto itr = values.begin() + 1; itr != values.end(); ++itr)
	{
		if (consistent != itr->size())throw std::length_error("(vector copy) Inconsistent row counts in container");
	}
#endif
	deepCopy(values);
}

ParallelMatrix::ParallelMatrix(const ParallelMatrix& cp) : ParallelMatrix()
{
	deepCopy(cp);
}

ParallelMatrix::~ParallelMatrix()
{
	if (data != nullptr)
	{
#if DEBUG_MATRIX == 1
		std::cout << "\tDeleting ParallelMatrix\n";
#endif
		delete[] data;//RAII
	}
#if DEBUG_MATRIX == 1
	else std::cout << "\tNo Deletion ParallelMatrix\n";
#endif
	data = nullptr;
}

ParallelMatrix& ParallelMatrix::operator=(const ParallelMatrix& cp)
{
	if (this != &cp)
	{
		deepCopy(cp, false);
	}
	return *this;
}

ParallelMatrix& ParallelMatrix::operator=(const std::vector<std::vector<float>>& values)
{
#if DEBUG_MATRIX >= 1
	if ((values.size() * values[0].size()) == 0)throw std::length_error("(vector =) Cannot have a matrix with zero elements");
	size_t consistent = values[0].size();
	for (auto itr = values.begin() + 1; itr != values.end(); ++itr)
	{
		if (consistent != itr->size())throw std::length_error("(vector =) Inconsistent row counts in container");
	}
#endif
	deepCopy(values, false);
	return *this;
}

ParallelMatrix ParallelMatrix::operator*(const ParallelMatrix& rhs) const
{
	if (columns != rhs.rows)
	{
#if DEBUG_MATRIX == 1
		std::cout << "Matrices do not align\n";
#endif
		return ParallelMatrix();
	}
	ParallelMatrix temp(rows, rhs.columns);
	//Going to avoid divisions here with a simple conditional
	unsigned int offset = 0;
	unsigned int curRow = 0;
	for (unsigned int i = 0; offset < temp.capacity; ++i)
	{
		if (i == temp.columns)
		{
			i = 0;
			++curRow;
		}
		float sum = 0;
		unsigned int rowOffset = curRow * this->columns;
#if DEBUG_MATRIX == 1
		std::cout << "Dot Product indices\n";
#endif
		for (unsigned int j = 0; j < this->columns; ++j)
		{
			sum += data[j + rowOffset] * rhs.data[j * rhs.columns + i];
#if DEBUG_MATRIX == 1
			std::cout << "product: " << (j + rowOffset) << " " << (j * rhs.columns + i) << "\n";
#endif
		}
		temp.data[offset++] = sum;
	}
	return temp;
}

unsigned int ParallelMatrix::getCapacity() const
{
	return capacity;
}

std::pair<unsigned int, unsigned int> ParallelMatrix::getDimensions() const
{
	return std::pair<unsigned int, unsigned int>(rows, columns);
}

std::string ParallelMatrix::getInfo() const
{
	if (data == nullptr)return "No Data Stored";
	std::string temp = "[";
	//To Mod or not to Mod
	for (unsigned int i = 0; i < capacity; ++i)
	{
		if (i % columns == 0)
		{
			if (i != 0) temp += '\n';
		}
		else temp += ",";
		temp += std::to_string(data[i]);
	}
	temp += ']';
	return temp;
}

ParallelMatrix ParallelMatrix::getParallelResult()
{
	return mC;
}

// For doing in place parallel arithmetic operations
void ParallelMatrix::setParallelMatrixOps(ParallelMatrix& matA)
{
	mA = &matA;
}

void ParallelMatrix::setParallelMatrixOps(ParallelMatrix& matA, ParallelMatrix& matB, bool multiplication)
{
	mA = &matA;
	mB = &matB;
	if (multiplication)
	{
		mC = ParallelMatrix(matA.rows, matB.columns);
		//TODO separate this case in final code
		//	Will use MACRO to separate the parallel Case A, B, and C respectively
		//	For now, need all compiled together to test in the shared main unit testing environment
		//	Some materials are going to be required based on current plans for the logbase 2 case:
		//		M = mA.rows * mB.columns (will need to map the final result of this to mC)
		//			M is the total number of dot products to perform
		//		N = mA.columns || mB.columns both are the same value
		//			N is the number of components in the dot products
		//		mD matrix will be sized with ((N+1) >> 1) elements in a row (i.e., columns)
		//			and 3 of these rows per dot product: ensuring the halving mechanism works per
		//			dispatch (where the initial dispatch requires N components -- i.e., first two rows 
		//			-- because the first dispatch merely multiplies componentwise before summing pairs)
		//			Therefore, because there are also M dot products to perform, mD is sized as follows:
		//				3*M by ((N + 1) >> 1)
		//		rI is 3*((N + 1) >> 1) so that we have the offset to the next dot product rows
		//			Note, after the multiplication, the summation stored will alternative between the 0th
		//			and 2ndth row within these 3 row chunks, acting as a double buffer between dispatches.
		//		cJ is 2*((N + 1) >> 1) to act as the alternating offset value between the local 3 row 
		//			chunk's 0th and 2nd row. In the first summation dispatch, this offset goes to the 
		//			index of the summation result, and alternates with the terms to sum
		//				I.E., odd cycles the offset goes to setter index value, even cycles to getter
		//				index
		//		curN is the current N value for the number of jobs dispatched. This is needed to 
		//			determine when there is not an additional pair for the current job to be summed with,
		//			in which case it merely stores the value from the getter location to the setter 
		//			location
		//		curItr is what determines which buffer in the three row chunk the data is gathered vs set
		M = mA->rows * mB->columns;
		N = mA->columns;
		unsigned int halved = (N + 1) >> 1;
		mD = ParallelMatrix(3 * M, halved);
		rI = 3 * halved;
		cJ = 2 * halved;
		curN = N;
		curItr = 0;
#if DEBUG_MATRIX == 2
		multiplications.store(0, std::memory_order_relaxed);
#endif
		//Object for the atomic testing
		atom = std::move(AtomicMatrix(matA.rows, matB.columns));
		//Third test for case C atomics
		intA = matA;
		intB = matB;
	}
}

void ParallelMatrix::parallelTrialA(std::mutex& m, unsigned int component)
{
	//First, need to determine which column and row the component index is in.
	// So, need both the dividend and remainder..
	unsigned int curRow = component / mC.columns, curColumn = component % mC.columns;

	ParallelMatrix& A = *mA;
	ParallelMatrix& B = *mB;
	float sum = 0;
	unsigned int rowOffset = curRow * A.columns;
	for (unsigned int j = 0; j < A.columns; ++j)
	{
		sum += A.data[j + rowOffset] * B.data[j * B.columns + curColumn];
	}
	mC.data[component] = sum;
}
#if DEBUG_MATRIX == 2
ParallelMatrix ParallelMatrix::getMD()
{
	return mD;
}
#endif

void ParallelMatrix::parallelTrialB0(std::mutex& m, unsigned int component)
{
	//This method is the starting point of a logbase2 approach to performing matrix
	//	multiplication in parallel -> i.e., a series of parallel dot products.
	//	The first stage is merely component-wise multiplication for the dot product
	//		and storing that result in a temporary buffer
	ParallelMatrix& A = *mA;
	ParallelMatrix& B = *mB;
	unsigned int curChunk = 0; // per iteration up to M below, this += rI
	unsigned int resetB = component * B.columns;// 0;
	unsigned int indexA = component;
	unsigned int indexB = resetB;// component;
	unsigned int lengthB = resetB + B.columns;
	unsigned int offsetA = A.columns;
	for (unsigned int i = 0; i < M; ++i)
	{
#if DEBUG_MATRIX == 2
		unsigned int tempA = A.data[indexA], tempB = B.data[indexB];
#endif
		mD.data[curChunk + component] = A.data[indexA] * B.data[indexB];
		curChunk += rI;
		if (++indexB == lengthB)
		{
			indexB = resetB;
			indexA += offsetA;
		}
#if DEBUG_MATRIX == 2
		multiplications.fetch_add(1, std::memory_order_relaxed);
		std::unique_lock<std::mutex> lock(m);
		std::cout << "components: " << component << " " << (curChunk + component) << " " << tempA << " " << tempB << "\n";
#endif
	}
}

void ParallelMatrix::setTrialB1(unsigned int count, bool reset)
{
	//count must be the number of active values to sum
	//	not the thread count!
	switch (curItr)
	{
	case 0:curItr = 1; break;
	case 1:curItr = 0; break;
	}
	if (reset)curItr = 1;
	curN = count - 1;
}

void ParallelMatrix::parallelTrialB1(std::mutex& m, unsigned int component)
{
	//Addition process will take values from the last buffer in the chunk of dot products
	//	and sum them together to be stored in the alternate buffer location
	unsigned int j = component;
	unsigned int g = component * 2;
	//curItr 1 means the data to gather is at the 0th row of the chunks
	if (g == curN)
	{
#if DEBUG_MATRIX == 2
		{
			std::lock_guard<std::mutex> lock(m);
			std::cout << "deadend component: " << component << " J: " << j << " G: " << g << " curN: " << curN << " curItr: " << curItr << "\n";
		}
#endif
		if (curItr == 1) j += cJ;
		else g += cJ;
#if DEBUG_MATRIX == 2
		{
			std::lock_guard<std::mutex> lock(m);
			std::cout << "de: " << component << " index g, j: " << g << " " << j << "\n";
		}
#endif
		for (unsigned int i = 0; i < M; ++i)
		{
			mD.data[j] = mD.data[g];
			j += rI;
			g += rI;
		}
	}
	else
	{
#if DEBUG_MATRIX == 2
		{
			std::lock_guard<std::mutex> lock(m);
			std::cout << "pair component: " << component << " J: " << j << " G: " << g << " curN: " << curN << " curItr: " << curItr << "\n";
		}
#endif
		if (curItr == 1) j += cJ;
		else g += cJ;
#if DEBUG_MATRIX == 2
		{
			std::lock_guard<std::mutex> lock(m);
			std::cout << "pc: " << component << " index g, j: " << g << " " << j << "\n";
		}
#endif
		for (unsigned int i = 0; i < M; ++i)
		{
#if DEBUG_MATRIX == 2
			{
				std::lock_guard<std::mutex> lock(m);
				std::cout << "blah: " << component << " j: " << j << " g: " << g << "\n";
				std::cout << "\t" << mD.data[g] << " " << mD.data[g + 1] << "\n";
			}
#endif
			mD.data[j] = mD.data[g] + mD.data[g + 1];
			j += rI;
			g += rI;
		}
	}
}

void ParallelMatrix::parallelTrialC0(std::mutex& m, unsigned int component)
{
	//This method is the starting point of a logbase2 approach to performing matrix
	//	multiplication in parallel -> i.e., a series of parallel dot products.
	//	The first stage is merely component-wise multiplication for the dot product
	//		and storing that result in a temporary buffer
	ParallelMatrix& A = *mA;
	ParallelMatrix& B = *mB;
	unsigned int curDot = 0; // per iteration up to M below
	unsigned int resetB = component * B.columns;// 0;
	unsigned int indexA = component;
	unsigned int indexB = resetB;// component;
	unsigned int lengthB = resetB + B.columns;
	unsigned int offsetA = A.columns;
	for (unsigned int i = 0; i < M; ++i)
	{
		atom.data[curDot].fetch_add(static_cast<int32_t>(AtomicMatrix::scale * A.data[indexA] * B.data[indexB]), std::memory_order_relaxed);
		++curDot;
		if (++indexB == lengthB)
		{
			indexB = resetB;
			indexA += offsetA;
		}
	}
}

void ParallelMatrix::parallelTrialC0Alt(std::mutex& m, unsigned int component)
{
	ParallelMatrix& A = *mA;
	ParallelMatrix& B = *mB;
	unsigned int dotSize = A.columns;
	unsigned int c = component / dotSize;
	unsigned int tModS = component - (c * dotSize);// component% dotSize;
	unsigned int cDiv = (c / dotSize);
	unsigned int cMod = c - (cDiv * dotSize);
	unsigned int indexA = cDiv * dotSize + tModS;// (c / dotSize)* dotSize + tModS;
	unsigned int indexB = cMod + tModS * dotSize;// (c % dotSize) + tModS * dotSize;
	atom.data[c].fetch_add(static_cast<int32_t>(AtomicMatrix::scale * A.data[indexA] * B.data[indexB]), std::memory_order_relaxed);
}

void ParallelMatrix::parallelTrialC0AltAlt(std::mutex& m, unsigned int component)
{
	unsigned int dotSize = mA->columns;
	unsigned int c = component / dotSize;
	unsigned int tModS = component - (c * dotSize);// component% dotSize;
	unsigned int cDiv = (c / dotSize);
	unsigned int cMod = c - (cDiv * dotSize);
	unsigned int indexA = cDiv * dotSize + tModS;// (c / dotSize)* dotSize + tModS;
	unsigned int indexB = cMod + tModS * dotSize;// (c % dotSize) + tModS * dotSize;
	atom.data[c].fetch_add(intA.data[indexA] * intB.data[indexB], std::memory_order_relaxed);
}

//Last method would be to give the threads a memory
//	They would perform the test to see if their current c value is equal to the last
//	If so, then they sum into another memory pool rather than the atomic add
//	If not, and the prior was valid, they perform the atomic add and reset the other
//		indexing values.
//	When working within the same dot product summation, the indexing values can follow
//		the serial loop style updates because the thread is guaranteed to have a contiguous
//		set of task indices it is working on.
//	This final test will likely need to be appended into the more formal looking final builds
//		of each respective Matrix Class because it will involve modification of the parallel
//		dispatch in the ThreadPool class.

void ParallelMatrix::parallelTrialC1(std::mutex& m, unsigned int component)
{
	mC.data[component] = AtomicMatrix::invScale * static_cast<float>(atom.data[component].load());
}

void ParallelMatrix::parallelTrialB2(std::mutex& m, unsigned int component)
{
	//curItr of 1 means the data was last stored in the 3rd row of the chunks
	//The final value is in the first index of the respective chunk row
	//The chunks are bijectively mapped by row major order to the matrix class
	unsigned int offset = (curItr == 1) ? cJ : 0;
	mC.data[component] = mD.data[component * rI + offset];
}

void ParallelMatrix::deepCopy(const ParallelMatrix& cp, bool allocate)
{
	bool cleanStart = true;
	if (allocate || data == nullptr) {}
	else
	{
		if (capacity == cp.capacity)
		{
			cleanStart = false;
		}
	}
	rows = cp.rows;
	columns = cp.columns;
	capacity = cp.capacity;
	if (cleanStart)
	{
#if DEBUG_MATRIX == 1
		std::cout << "DEBUG (deepCopy PMat): Deleting old Matrix Data\n";
#endif
		delete[] data;
		data = new (std::nothrow)float[capacity];
#if DEBUG_MATRIX >= 1
		if (data == nullptr)std::cout << "PMAT DEEPCOPY PMAT ALLOCATION FAILURE!\n";
#endif
	}
#if DEBUG_MATRIX == 1
	else
		std::cout << "DEBUG (deepCopy PMat): Repurposing Matrix Data\n";
#endif
	for (unsigned int i = 0; i < capacity; ++i)
	{
		data[i] = cp.data[i];
	}
}

void ParallelMatrix::deepCopy(const std::vector<std::vector<float>>& values, bool allocate)
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
	rows = static_cast<unsigned int>(values.size());
	columns = static_cast<unsigned int>(values[0].size());
	capacity = rows * columns;
	if (cleanStart)
	{
#if DEBUG_MATRIX == 1
		std::cout << "DEBUG (deepCopy vector): Deleting old Matrix Data\n";
#endif
		delete[] data;
		data = new (std::nothrow)float[capacity];
#if DEBUG_MATRIX >= 1
		if (data == nullptr)std::cout << "PMAT DEEPCOPY VECTOR ALLOCATION FAILURE!\n";
#endif
	}
#if DEBUG_MATRIX == 1
	else
		std::cout << "DEBUG (deepCopy vector): Repurposing Matrix Data\n";
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

std::ostream& operator<<(std::ostream& out, const ParallelMatrix& mat)
{
	return out << mat.getInfo();
}

bool operator==(const ParallelMatrix& A, const ParallelMatrix& B)
{
	if (A.capacity != B.capacity)return false;
	for (unsigned int i = 0; i < A.capacity; ++i)
	{
		if (A.data[i] != B.data[i])return false;
	}
	return true;
}

//Templated case below
template <size_t T, size_t S>
BaseMatrix<T, S>::BaseMatrix()
{

}

template <size_t T, size_t S>
BaseMatrix<T, S>::~BaseMatrix()
{

}

template <size_t T, size_t S>
template <size_t V>
BaseMatrix<T, V> BaseMatrix<T, S>::operator*(BaseMatrix<S, V>& ref)
{

}