/*
Author: Dan Rehberg
Modified Date: 12/9/2022
*/
#include "SerialMatrix.hpp"

//Parallel Stuff
SerialMatrix* SerialMatrix::mA = nullptr;
SerialMatrix* SerialMatrix::mB = nullptr;
SerialMatrix SerialMatrix::mC = SerialMatrix(1, 1);

void SerialMatrix::setParallelMatrixOps(SerialMatrix& matA, SerialMatrix& matB, bool multiplication)
{
	mA = &matA;
	mB = &matB;
	if (multiplication)
	{
		mC = SerialMatrix(matA.rows, matB.columns);
	}
}

void SerialMatrix::parallelDotProducts(std::mutex& m, unsigned int component)
{
	//First, need to determine which column and row the component index is in.
	// So, need both the dividend and remainder..
	unsigned int curRow = component / mC.columns, curColumn = component % mC.columns;

	SerialMatrix& A = *mA;
	SerialMatrix& B = *mB;
	float sum = 0;
	unsigned int rowOffset = curRow * A.columns;
	for (unsigned int j = 0; j < A.columns; ++j)
	{
		sum += A.data[j + rowOffset] * B.data[j * B.columns + curColumn];
	}
	mC.data[component] = sum;
}

SerialMatrix&& SerialMatrix::moveParallelResult()
{
	return std::move(mC);
}

//End Parallel Stuff


SerialMatrix::SerialMatrix() : rows(0), columns(0), capacity(0), data(nullptr)
{

}

SerialMatrix::SerialMatrix(size_t r, size_t c) : SerialMatrix()
{
	rows = r;
	columns = c;
	capacity = r * c;
	if (capacity == 0)throw std::length_error("(r,c constructor) Cannot have a matrix with zero elements");
	data = new (std::nothrow)float[capacity];
	if (data == nullptr)throw std::bad_alloc();//"Serial Matrix class, bad allocation on Row Col Constructor"
}

SerialMatrix::SerialMatrix(const std::vector<std::vector<float>>& values) : SerialMatrix()
{
	if ((values.size() * values[0].size()) == 0)throw std::length_error("(vector copy) Cannot have a matrix with zero elements");
	size_t consistent = values[0].size();
	for (auto itr = values.begin() + 1; itr != values.end(); ++itr)
	{
		if (consistent != itr->size())throw std::length_error("(vector copy) Inconsistent row counts in container");
	}
	deepCopy(values);
}

SerialMatrix::SerialMatrix(const SerialMatrix& cp) : SerialMatrix()
{
	deepCopy(cp);
}

SerialMatrix::SerialMatrix(SerialMatrix&& rhs) : SerialMatrix()
{
	rows = rhs.rows;
	columns = rhs.columns;
	capacity = rhs.capacity;
	data = rhs.data;
	rhs.data = nullptr;
	rhs.capacity = 0;
	rhs.rows = 0;
	rhs.columns = 0;
}

SerialMatrix::~SerialMatrix()
{
	if (data != nullptr)
	{
		delete[] data;//RAII
	}
	data = nullptr;
}

SerialMatrix& SerialMatrix::operator=(const SerialMatrix& cp)
{
	if (this != &cp)
	{
		deepCopy(cp, false);
	}
	return *this;
}

SerialMatrix& SerialMatrix::operator=(SerialMatrix&& rhs)
{
	if (data != nullptr)delete[] data;
	rows = rhs.rows;
	columns = rhs.columns;
	capacity = rhs.capacity;
	data = rhs.data;
	rhs.data = nullptr;
	rhs.capacity = 0;
	rhs.rows = 0;
	rhs.columns = 0;
	return *this;
}

SerialMatrix& SerialMatrix::operator=(const std::vector<std::vector<float>>& values)
{
	if ((values.size() * values[0].size()) == 0)throw std::length_error("(vector =) Cannot have a matrix with zero elements");
	size_t consistent = values[0].size();
	for (auto itr = values.begin() + 1; itr != values.end(); ++itr)
	{
		if (consistent != itr->size())throw std::length_error("(vector =) Inconsistent row counts in container");
	}
	deepCopy(values, false);
	return *this;
}

SerialMatrix SerialMatrix::operator*(const SerialMatrix& rhs) const
{
	if (columns != rhs.rows)
	{
		return SerialMatrix();
	}
	SerialMatrix temp(rows, rhs.columns);
	//Going to avoid divisions here with a simple conditional
	size_t offset = 0;
	size_t curRow = 0;
	for (size_t i = 0; offset < temp.capacity; ++i)
	{
		if (i == temp.columns)
		{
			i = 0;
			++curRow;
		}
		float sum = 0;
		size_t rowOffset = curRow * this->columns;
		for (size_t j = 0; j < this->columns; ++j)
		{
			sum += data[j + rowOffset] * rhs.data[j * rhs.columns + i];
		}
		temp.data[offset++] = sum;
	}
	return temp;
}

SerialMatrix SerialMatrix::operator-(const SerialMatrix& rhs) const
{
	SerialMatrix temp(rows, columns);
	//row or columns modifications only -- might extend, needed to run the NN for now
	if (rhs.rows == 1)
	{
		if (rhs.columns != this->columns)
			throw std::range_error(R"(Right matrix with 1 row must match column size with left
				matrix\n\t right matrix column size is: )" + std::to_string(rhs.columns));
		size_t curIndex = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < columns; ++j) //[[Potential Optimization Pending]]
			{
				temp.data[curIndex] = data[curIndex] - rhs.data[j];
				++curIndex;
			}
		}
	}
	/*else if (rhs.columns == 1) //Not necessary for minimal viable product..
	{
		if (rhs.rows != this->rows)
			throw std::range_error(R"(Right matrix with 1 column must match row size with left
				matrix\n\t right matrix row size is: )" + std::to_string(rhs.rows));
	}*/
	else if (rhs.capacity == this->capacity)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			temp.data[i] = data[i] - rhs.data[i];
		}
	}
	else throw std::range_error(R"(Right matrix needs like component counts total, 
		or 1 row with same columns, 
		or 1 column with same rows
		\n\t right matrix has dimensions: )" + std::to_string(rhs.rows) + ", " +
		std::to_string(rhs.columns));
	return temp;
}

SerialMatrix SerialMatrix::operator/(const SerialMatrix& rhs) const
{
	SerialMatrix temp(rows, columns);
	//row or columns modifications only -- might extend, needed to run the NN for now
	if (rhs.rows == 1)
	{
		if (rhs.columns != this->columns)
			throw std::range_error(R"(Right matrix with 1 row must match column size with left
				matrix\n\t right matrix column size is: )" + std::to_string(rhs.columns));
		size_t curIndex = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < columns; ++j) //[[Potential Optimization Pending]]
			{
				temp.data[curIndex] = data[curIndex] / rhs.data[j];
				++curIndex;
			}
		}
	}
	/*else if (rhs.columns == 1) //Not necessary for minimal viable product..
	{
		if (rhs.rows != this->rows)
			throw std::range_error(R"(Right matrix with 1 column must match row size with left
				matrix\n\t right matrix row size is: )" + std::to_string(rhs.rows));
	}*/
	else if (rhs.capacity == this->capacity)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			temp.data[i] = data[i] / rhs.data[i];
		}
	}
	else throw std::range_error(R"(Right matrix needs like component counts total, 
		or 1 row with same columns, 
		or 1 column with same rows
		\n\t right matrix has dimensions: )" + std::to_string(rhs.rows) + ", " +
		std::to_string(rhs.columns));
	return temp;
}

SerialMatrix SerialMatrix::operator+(const SerialMatrix& rhs) const
{
	SerialMatrix temp(rows, columns);
	//row or columns modifications only -- might extend, needed to run the NN for now
	if (rhs.rows == 1)
	{
		if (rhs.columns != this->columns)
			throw std::range_error(R"(Right matrix with 1 row must match column size with left
				matrix\n\t right matrix column size is: )" + std::to_string(rhs.columns));
		size_t curIndex = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < columns; ++j) //[[Potential Optimization Pending]]
			{
				temp.data[curIndex] = data[curIndex] + rhs.data[j];
				++curIndex;
			}
		}
	}
	/*else if (rhs.columns == 1) //Not necessary for minimal viable product..
	{
		if (rhs.rows != this->rows)
			throw std::range_error(R"(Right matrix with 1 column must match row size with left
				matrix\n\t right matrix row size is: )" + std::to_string(rhs.rows));
	}*/
	else if (rhs.capacity == this->capacity)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			temp.data[i] = data[i] + rhs.data[i];
		}
	}
	else throw std::range_error(R"(Right matrix needs like component counts total, 
		or 1 row with same columns, 
		or 1 column with same rows
		\n\t right matrix has dimensions: )" + std::to_string(rhs.rows) + ", " +
		std::to_string(rhs.columns));
	return temp;
}

SerialMatrix& SerialMatrix::operator+=(const SerialMatrix& rhs)
{
	//row or columns modifications only -- might extend, needed to run the NN for now
	if (rhs.rows == 1)
	{
		if (rhs.columns != this->columns)
			throw std::range_error(R"(Right matrix with 1 row must match column size with left
				matrix\n\t right matrix column size is: )" + std::to_string(rhs.columns));
		size_t curIndex = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < columns; ++j) //[[Potential Optimization Pending]]
			{
				data[curIndex++] += rhs.data[j];
			}
		}
	}
	/*else if (rhs.columns == 1) //Not necessary for minimal viable product..
	{
		if (rhs.rows != this->rows)
			throw std::range_error(R"(Right matrix with 1 column must match row size with left
				matrix\n\t right matrix row size is: )" + std::to_string(rhs.rows));
	}*/
	else if (rhs.capacity == this->capacity)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			data[i] += rhs.data[i];
		}
	}
	else throw std::range_error(R"(Right matrix needs like component counts total, 
		or 1 row with same columns, 
		or 1 column with same rows
		\n\t right matrix has dimensions: )" + std::to_string(rhs.rows) + ", " +
		std::to_string(rhs.columns));
	return *this;
}

SerialMatrix SerialMatrix::transpose(const SerialMatrix& ref, size_t rowStart)
{
	if (rowStart > ref.rows)throw std::range_error(R"(Row start must be less than row count 
		for a reduced Transpose operation: matrix to transpose has: )"
		+ std::to_string(ref.rows) + " but rowStart was set to: " + std::to_string(rowStart));
	size_t col = ref.rows - rowStart;
	SerialMatrix temp(ref.columns, col);
	size_t offset = rowStart * ref.columns;
	size_t cur = 0;
	for (size_t i = 0; i < ref.columns; ++i)
	{
		size_t index = i;
		for (size_t j = rowStart; j < ref.rows; ++j)
		{
			temp.data[cur] = ref.data[index + offset];
			index += ref.columns;
			++cur;
		}
	}
	return temp;
}

SerialMatrix SerialMatrix::componentwise(const SerialMatrix& A, const SerialMatrix& B)
{
	if (A.capacity != B.capacity)throw std::length_error(R"(Componentwise multiplication of 
		matrices requires the matrices have the same total cardinality;
		A has: )" + std::to_string(A.capacity) + " while B has: " + std::to_string(B.capacity));
	SerialMatrix temp(A.rows, A.columns);
	for (size_t i = 0; i < A.capacity; ++i) temp.data[i] = A.data[i] * B.data[i];
	return temp;
}

size_t SerialMatrix::getCapacity() const
{
	return capacity;
}

std::pair<size_t, size_t> SerialMatrix::getDimensions() const
{
	return std::pair<size_t, size_t>(rows, columns);
}

std::string SerialMatrix::getInfo() const
{
	if (data == nullptr)return "No Data Stored";
	std::string temp = "[";
	//To Mod or not to Mod
	for (size_t i = 0; i < capacity; ++i)
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

float SerialMatrix::mean(const SerialMatrix& ref)
{
	float temp = 0.0f;
	for (size_t i = 0; i < ref.capacity; ++i)
	{
		temp += ref.data[i];
	}
	return temp / static_cast<float>(ref.capacity);
}

void SerialMatrix::activateTanH()
{
	for (size_t i = 0; i < capacity; ++i)
	{
		data[i] = std::tanh(data[i]);
	}
}

SerialMatrix SerialMatrix::mean(const SerialMatrix& ref, bool row)
{
	if (row)
	{
		SerialMatrix temp(1, ref.rows);
		float inv = 1.0f / static_cast<float>(ref.columns);
		for (size_t i = 0; i < ref.rows; ++i)
		{
			float cur = 0.0f;
			size_t curIndex = 0;
			for (size_t j = 0; j < ref.columns; ++j)
			{
				cur += ref.data[curIndex++];
			}
			temp.data[i] = cur * inv;
		}
		return temp;
	}
	else
	{
		SerialMatrix temp(1, ref.columns);
		float inv = 1.0f / static_cast<float>(ref.rows);
		for (size_t i = 0; i < ref.columns; ++i)
		{
			float cur = 0.0f;
			size_t curIndex = 0;
			for (size_t j = 0; j < ref.rows; ++j)
			{
				cur += ref.data[curIndex];
				curIndex += ref.columns;
			}
			temp.data[i] = cur * inv;
		}
		return temp;
	}
}

SerialMatrix SerialMatrix::standardDeviations(const SerialMatrix& ref, bool row)
{
	if (row)
	{
		SerialMatrix temp = SerialMatrix::mean(ref, row);
		for (size_t i = 0; i < ref.rows; ++i)
		{
			float squareMeanDifference = 0.0f;
			size_t curIndex = 0;
			for (size_t j = 0; j < ref.columns; ++j)
			{
				float diff = ref.data[curIndex++] - temp.data[i];
				squareMeanDifference = diff * diff;
			}
			temp.data[i] = std::sqrt(squareMeanDifference / ref.columns);
		}
		return temp;
	}
	else
	{
		SerialMatrix temp = SerialMatrix::mean(ref, row);
		for (size_t i = 0; i < ref.columns; ++i)
		{
			float squareMeanDifference = 0.0f;
			size_t curIndex = 0;
			for (size_t j = 0; j < ref.rows; ++j)
			{
				float diff = ref.data[curIndex] - temp.data[i];
				squareMeanDifference += diff * diff;
				curIndex += ref.columns;
			}
			temp.data[i] = std::sqrt(squareMeanDifference / ref.rows);
		}
		return temp;
	}
}

SerialMatrix SerialMatrix::square(const SerialMatrix& ref)
{
	SerialMatrix temp(ref.rows, ref.columns);
	for (size_t i = 0; i < ref.capacity; ++i)
	{
		temp.data[i] = ref.data[i] * ref.data[i];
	}
	return temp;
}

//Do not need all of the features from other linear algebra libraries
//	Just including what is necessary to run the Neural Network class.
SerialMatrix SerialMatrix::addOnes(const SerialMatrix& ref)
{
	SerialMatrix temp(ref.rows, ref.columns + 1);
	size_t columnCount = 0;// ref.columns;
	size_t curIndex = 0;
	for (size_t i = 0; i < ref.capacity; ++i)
	{
		if (columnCount == 0)
		{
			temp.data[curIndex] = 1.0f;
			++curIndex;
			columnCount = ref.columns;
		}
		temp.data[curIndex] = ref.data[i];
		++curIndex;
		--columnCount;
	}
	return temp;
}

void SerialMatrix::deepCopy(const SerialMatrix& cp, bool allocate)
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
		delete[] data;
		data = new (std::nothrow)float[capacity];
		if (data == nullptr)throw std::bad_alloc();//"Serial Matrix class, bad allocation on Matrix DeepCopy"
	}
	for (size_t i = 0; i < capacity; ++i)
	{
		data[i] = cp.data[i];
	}
}

void SerialMatrix::deepCopy(const std::vector<std::vector<float>>& values, bool allocate)
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
	rows = static_cast<size_t>(values.size());
	columns = static_cast<size_t>(values[0].size());
	capacity = rows * columns;
	if (cleanStart)
	{
		delete[] data;
		data = new (std::nothrow)float[capacity];
		if (data == nullptr)throw std::bad_alloc();//"Serial Matrix class, bad allocation on STD::Vector DeepCopy"
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

std::ostream& operator<<(std::ostream& out, const SerialMatrix& mat)
{
	return out << mat.getInfo();
}

bool operator==(const SerialMatrix& A, const SerialMatrix& B)
{
	if (A.capacity != B.capacity)return false;
	for (size_t i = 0; i < A.capacity; ++i)
	{
		if (A.data[i] != B.data[i])return false;
	}
	return true;
}

SerialMatrix operator-(const float lhs, const SerialMatrix& rhs)
{
	SerialMatrix temp(rhs.rows, rhs.columns);
	for (size_t i = 0; i < rhs.capacity; ++i)
	{
		temp.data[i] = lhs - rhs.data[i];
	}
	return temp;
}

SerialMatrix operator/(const float lhs, const SerialMatrix& rhs)
{
	float inv = 1.0f / lhs;
	return inv * rhs;
}

SerialMatrix operator*(const float lhs, const SerialMatrix& rhs)
{
	SerialMatrix temp(rhs.rows, rhs.columns);
	for (size_t i = 0; i < rhs.capacity; ++i)
	{
		temp.data[i] = lhs * rhs.data[i];
	}
	return temp;
}