/*
Author: Dan Rehberg
Modified Date: 1/3/2023
*/
#include "IntAtomicMatrix.hpp"

//static data values
IntegerMatrix* IntegerMatrix::mA = nullptr;
IntegerMatrix* IntegerMatrix::mB = nullptr;
IntegerMatrix IntegerMatrix::mC = IntegerMatrix(1, 1);
AtomicMatrix IntegerMatrix::atom = std::move(AtomicMatrix(1, 1));
unsigned int IntegerMatrix::atomSize = 0;

IntegerMatrix::IntegerMatrix() : rows(0), columns(0), capacity(0), data(nullptr)
{

}

IntegerMatrix::IntegerMatrix(unsigned int r, unsigned int c) : IntegerMatrix()
{
	rows = r;
	columns = c;
	capacity = r * c;
#if DEBUG_MATRIX >= 1
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
#endif
	data = new (std::nothrow)int_fast64_t[capacity];
#if DEBUG_MATRIX >= 1
	if (data == nullptr)std::cout << "PMAT CONSTRUCTOR ALLOCATION FAILURE!\n";
#endif
}

IntegerMatrix::IntegerMatrix(const std::vector<std::vector<float>>& values) : IntegerMatrix()
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

IntegerMatrix::IntegerMatrix(const IntegerMatrix& cp) : IntegerMatrix()
{
	deepCopy(cp);
}

IntegerMatrix::~IntegerMatrix()
{
	if (data != nullptr)
	{
#if DEBUG_MATRIX == 1
		std::cout << "\tDeleting IntegerMatrix\n";
#endif
		delete[] data;//RAII
	}
#if DEBUG_MATRIX == 1
	else std::cout << "\tNo Deletion IntegerMatrix\n";
#endif
	data = nullptr;
}

IntegerMatrix& IntegerMatrix::operator=(const IntegerMatrix& cp)
{
	if (this != &cp)
	{
		deepCopy(cp, false);
	}
	return *this;
}

IntegerMatrix& IntegerMatrix::operator=(const std::vector<std::vector<float>>& values)
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

IntegerMatrix IntegerMatrix::operator*(const IntegerMatrix& rhs) const
{
	if (columns != rhs.rows)
	{
#if DEBUG_MATRIX == 1
		std::cout << "Matrices do not align\n";
#endif
		return IntegerMatrix();
	}
	IntegerMatrix temp(rows, rhs.columns);
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

unsigned int IntegerMatrix::getCapacity() const
{
	return capacity;
}

std::pair<unsigned int, unsigned int> IntegerMatrix::getDimensions() const
{
	return std::pair<unsigned int, unsigned int>(rows, columns);
}

std::string IntegerMatrix::getInfo() const
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
		temp += std::to_string(AtomicMatrix::scale * static_cast<float>(data[i]));
	}
	temp += ']';
	return temp;
}

IntegerMatrix IntegerMatrix::getParallelResult()
{
	return mC;
}

// For doing in place parallel arithmetic operations
void IntegerMatrix::setIntegerMatrixOps(IntegerMatrix& matA)
{
	mA = &matA;
}

void IntegerMatrix::setParallelMatrixOps(IntegerMatrix& matA, IntegerMatrix& matB, bool multiplication)
{
	mA = &matA;
	mB = &matB;
	if (multiplication)
	{
		mC = IntegerMatrix(matA.rows, matB.columns);
		//Object for the atomic testing
		atom = std::move(AtomicMatrix(matA.rows, matB.columns));
		atomSize = matA.rows * matB.columns;
	}
}

/*void IntegerMatrix::parallelTrialC0AltAlt(MemoryTest& mem, std::mutex& m, unsigned int component)
{
	IntegerMatrix& A = *mA;
	IntegerMatrix& B = *mB;
	unsigned int dotSize = A.columns;
	unsigned int c = component / dotSize;
	unsigned int tModS = component - (c * dotSize);// component% dotSize;
	unsigned int cDiv = (c / dotSize);
	unsigned int cMod = c - (cDiv * dotSize);
	unsigned int indexA = cDiv * dotSize + tModS;// (c / dotSize)* dotSize + tModS;
	unsigned int indexB = cMod + tModS * dotSize;// (c % dotSize) + tModS * dotSize;
	if (c != mem.curJob)
	{
		if (mem.curJob >= 0)
			atom.data[c].fetch_add(mem.memory, std::memory_order_relaxed);
		mem.curJob = c;
		mem.memory = 0;
	}
	mem.memory += mA->data[indexA] * mB->data[indexB];
}*/

void IntegerMatrix::parallelTrialC0AltAlt(std::mutex& m, unsigned int start, unsigned int end)
{
	IntegerMatrix& A = *mA;
	IntegerMatrix& B = *mB;
	unsigned int dotSize = A.columns;//i.e., how many elements in a row
	unsigned int bColumns = B.columns;
	unsigned int c = start / dotSize;
	unsigned int tModS = start - (c * dotSize);
	unsigned int cDiv = (c / dotSize);
	unsigned int cMod = c - (cDiv * dotSize);
	unsigned int indexA = cDiv * dotSize + tModS;// (c / dotSize)* dotSize + tModS;
	unsigned int indexB = cMod + tModS * dotSize;// (c % dotSize) + tModS * dotSize;
	unsigned int cur = indexA % dotSize;
	int count = 0;
	for (unsigned int i = start; i < end; ++i)
	{
		++indexA;
		indexB += bColumns;
		if (++cur == dotSize)
		{
			atom.data[c].fetch_add(count, std::memory_order_relaxed);
			c = i / dotSize;
			count = 0;
			//indices have changed..
			tModS = i - (c * dotSize);
			cDiv = (c / dotSize);
			cMod = c - (cDiv * dotSize);
			indexA = cDiv * dotSize + tModS;
			indexB = cMod + tModS * dotSize;
			cur = indexA % dotSize;
		}

		count += A.data[indexA] * B.data[indexB];
	}
	atom.data[c].fetch_add(count, std::memory_order_relaxed);
}
void IntegerMatrix::parallelTrialC0AltAltAlt(std::mutex& m, unsigned int start, unsigned int end)
{
	IntegerMatrix& A = *mA;
	IntegerMatrix& B = *mB;
	unsigned int dotSize = A.columns;
	unsigned int c = start / dotSize;//index into the C data
	unsigned int baseA = c / B.columns;// *dotSize;//starting row index for A data
	unsigned int baseB = c - (baseA * dotSize);// c% B.columns;//starting column index for B data
	baseA *= dotSize;//Removed calculation in declaration to use data to avoid modulus above (no noticable performance change as it occurs once per thread..)
	unsigned int counter = start - (c * dotSize);//starting position in a dot product (reset per new dot product)
	unsigned int indexA = baseA + counter;
	unsigned int indexB = baseB + counter * B.columns;
	int sum = 0;
	for (unsigned int i = start; i < end; ++i)
	{
		sum += A.data[indexA] * B.data[indexB];
		if (++counter == dotSize)
		{
			counter = 0;
			if (++baseB == B.columns)
			{
				baseA += dotSize;
				baseB = 0;
			}
			indexA = baseA;
			indexB = baseB;
			atom.data[c++].fetch_add(sum, std::memory_order_relaxed);
			sum = 0;
		}
		else
		{
			++indexA;
			indexB += B.columns;
		}
	}
	if (c < atomSize)
	{
		atom.data[c].fetch_add(sum, std::memory_order_relaxed);
	}
}

void IntegerMatrix::deepCopy(const IntegerMatrix& cp, bool allocate)
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
		data = new (std::nothrow)int_fast64_t[capacity];
#if DEBUG_MATRIX >= 1
		if (data == nullptr)throw std::exception("PMAT DEEPCOPY PMAT ALLOCATION FAILURE!");
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

void IntegerMatrix::deepCopy(const std::vector<std::vector<float>>& values, bool allocate)
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
		data = new (std::nothrow)int_fast64_t[capacity];
#if DEBUG_MATRIX >= 1
		if (data == nullptr)throw std::exception("PMAT DEEPCOPY VECTOR ALLOCATION FAILURE!");
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
			data[(itr - values.begin()) * columns + (jtr - itr->begin())] =
				static_cast<int_fast64_t>(AtomicMatrix::scale * (*jtr));
		}
	}
}

std::ostream& operator<<(std::ostream& out, const IntegerMatrix& mat)
{
	return out << mat.getInfo();
}

bool operator==(const IntegerMatrix& A, const IntegerMatrix& B)
{
	if (A.capacity != B.capacity)return false;
	for (unsigned int i = 0; i < A.capacity; ++i)
	{
		if (A.data[i] != B.data[i])return false;
	}
	return true;
}
