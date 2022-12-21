/*
Author: Dan Rehberg
Modified Date: 12/20/2022
*/
#include "AtomicMatrix.hpp"

//static data values
FloatMatrix* FloatMatrix::mA = nullptr;
FloatMatrix* FloatMatrix::mB = nullptr;
FloatMatrix FloatMatrix::mC = FloatMatrix(1, 1);
AtomicMatrix FloatMatrix::atom = std::move(AtomicMatrix(1, 1));

AtomicMatrix::AtomicMatrix() : data(nullptr) {}

AtomicMatrix::AtomicMatrix(unsigned int rows, unsigned int columns) : AtomicMatrix()
{
	unsigned int capacity = rows * columns;
#if DEBUG_MATRIX >= 1
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
#endif
	data = new (std::nothrow)std::atomic<int_fast64_t>[capacity];
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

FloatMatrix::FloatMatrix() : rows(0), columns(0), capacity(0), data(nullptr)
{

}

FloatMatrix::FloatMatrix(unsigned int r, unsigned int c) : FloatMatrix()
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

FloatMatrix::FloatMatrix(const std::vector<std::vector<float>>& values) : FloatMatrix()
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

FloatMatrix::FloatMatrix(const FloatMatrix& cp) : FloatMatrix()
{
	deepCopy(cp);
}

FloatMatrix::~FloatMatrix()
{
	if (data != nullptr)
	{
#if DEBUG_MATRIX == 1
		std::cout << "\tDeleting FloatMatrix\n";
#endif
		delete[] data;//RAII
	}
#if DEBUG_MATRIX == 1
	else std::cout << "\tNo Deletion FloatMatrix\n";
#endif
	data = nullptr;
}

FloatMatrix& FloatMatrix::operator=(const FloatMatrix& cp)
{
	if (this != &cp)
	{
		deepCopy(cp, false);
	}
	return *this;
}

FloatMatrix& FloatMatrix::operator=(const std::vector<std::vector<float>>& values)
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

FloatMatrix FloatMatrix::operator*(const FloatMatrix& rhs) const
{
	if (columns != rhs.rows)
	{
#if DEBUG_MATRIX == 1
		std::cout << "Matrices do not align\n";
#endif
		return FloatMatrix();
	}
	FloatMatrix temp(rows, rhs.columns);
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

unsigned int FloatMatrix::getCapacity() const
{
	return capacity;
}

std::pair<unsigned int, unsigned int> FloatMatrix::getDimensions() const
{
	return std::pair<unsigned int, unsigned int>(rows, columns);
}

std::string FloatMatrix::getInfo() const
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

FloatMatrix FloatMatrix::getParallelResult()
{
	return mC;
}

// For doing in place parallel arithmetic operations
void FloatMatrix::setFloatMatrixOps(FloatMatrix& matA)
{
	mA = &matA;
}

void FloatMatrix::setParallelMatrixOps(FloatMatrix& matA, FloatMatrix& matB, bool multiplication)
{
	mA = &matA;
	mB = &matB;
	if (multiplication)
	{
		mC = FloatMatrix(matA.rows, matB.columns);
		//Object for the atomic testing
		atom = std::move(AtomicMatrix(matA.rows, matB.columns));
		//Third test for case C atomics
	}
}

/*void FloatMatrix::parallelTrialC0AltAlt(MemoryTest& mem, std::mutex& m, unsigned int component)
{
	FloatMatrix& A = *mA;
	FloatMatrix& B = *mB;
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

void FloatMatrix::parallelTrialC0AltAlt(std::mutex& m, unsigned int start, unsigned int end)
{
	FloatMatrix& A = *mA;
	FloatMatrix& B = *mB;
	unsigned int dotSize = A.columns;//i.e., how many elements in a row
	unsigned int bColumns = B.columns;
	unsigned int c = start / dotSize;
	unsigned int tModS = start - (c * dotSize);
	unsigned int cDiv = (c / dotSize);
	unsigned int cMod = c - (cDiv * dotSize);
	unsigned int indexA = cDiv * dotSize + tModS;// (c / dotSize)* dotSize + tModS;
	unsigned int indexB = cMod + tModS * dotSize;// (c % dotSize) + tModS * dotSize;
	unsigned int cur = indexA % dotSize;
	float count = 0.0f;
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
	atom.data[c].fetch_add(static_cast<int_fast64_t>(count), std::memory_order_relaxed);
}

void FloatMatrix::deepCopy(const FloatMatrix& cp, bool allocate)
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

void FloatMatrix::deepCopy(const std::vector<std::vector<float>>& values, bool allocate)
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
				*jtr;// static_cast<int>(AtomicMatrix::scale * (*jtr));
		}
	}
}

std::ostream& operator<<(std::ostream& out, const FloatMatrix& mat)
{
	return out << mat.getInfo();
}

bool operator==(const FloatMatrix& A, const FloatMatrix& B)
{
	if (A.capacity != B.capacity)return false;
	for (unsigned int i = 0; i < A.capacity; ++i)
	{
		if (A.data[i] != B.data[i])return false;
	}
	return true;
}
