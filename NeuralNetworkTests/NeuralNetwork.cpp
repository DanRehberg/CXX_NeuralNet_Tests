/*
Author: Dan Rehberg
Modified Date: 12/9/2022
*/
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() : xMean(1, 1), tMean(1, 1), xStd(1, 1), tStd(1, 1)
{
	input = 0;
	output = 0;
	epoch = 0;
}

NeuralNetwork::NeuralNetwork(const size_t inputCount, const std::vector<size_t>& hiddenCount,
	const size_t outputCount) : NeuralNetwork()
{
	input = inputCount;
	hidden.reserve(hiddenCount.size());
	for (auto itr = hiddenCount.begin(); itr != hiddenCount.end(); ++itr)
	{
		hidden.push_back(*itr);
	}
	output = outputCount;

	epoch = 0;

	//Build the weights
	weights.resize(hiddenCount.size() + 1);
	Z.resize(weights.size() + 1);
	//	But, also go ahead and size them appropriately
	size_t rowCount = inputCount + 1;
	for (size_t i = 0; i < hiddenCount.size(); ++i)
	{
		//ith index in hidden is the column count
		size_t columnCount = hiddenCount[i];
		//Note, could use this time to modify the matrix with random values...
		//	Could be as simple as building out a vector of vectors to instantiate the 
		//		matrix object with.
		weights[i] = Matrix(rowCount, columnCount);
		rowCount = columnCount + 1;
	}
	//Ouput layer weights
	weights.back() = Matrix(rowCount, outputCount);
}

NeuralNetwork::~NeuralNetwork()
{
	//If the weights were manually allocated, then delete here
	//Seems reasonable for vector in this case (no need for the low-level control)
}

void NeuralNetwork::testWeights()
{
	float min = -0.01f, max = 0.01f;
	float range = max - min;
	for (auto itr = weights.begin(); itr != weights.end(); ++itr)
	{
		std::vector<std::vector<float>> testValues;
		std::vector<float> rowValues;
		size_t count = 0;
		float increment = range / static_cast<float>(itr->getCapacity() - 1);
		for (size_t i = 0; i < itr->getCapacity(); ++i)
		{
			if (count == itr->getDimensions().second)
			{
				testValues.push_back(rowValues);
				count = 0;
				rowValues.clear();
			}
			if (itr != (weights.end() - 1))
			{
				rowValues.push_back(min + (static_cast<float>(i) * increment));
			}
			else rowValues.push_back(0.0f);
			++count;
		}
		testValues.push_back(rowValues);
		if (itr != (weights.end() - 1))
		{
			for (size_t i = 0; i < rowValues.size(); ++i)
			{
				float offset = (static_cast<float>(i) - static_cast<float>(rowValues.size()) * 0.5f) * 0.2f;
				for (size_t j = 0; j < testValues.size(); ++j)
				{
					testValues[j][i] += offset;
				}
			}
		}
		(*itr) = Matrix(testValues);
	}
}

std::string NeuralNetwork::getInfo() const
{
	std::string temp = "Network values for inputs, hidden, and output: " +
		std::to_string(input) + ", [";
	for (size_t i = 0; i < hidden.size(); ++i)
	{
		temp += std::to_string(hidden[i]);
		if (i != hidden.size())temp += ", ";
	}
	temp += "], " + std::to_string(output) + "\n\t\t Weight Dimensions:";
	for (size_t i = 0; i < weights.size(); ++i)
	{
		temp += "\n\t\t\t" + std::to_string(weights[i].getDimensions().first) +
			", " + std::to_string(weights[i].getDimensions().second);
	}

	return temp;
}

void NeuralNetwork::train(Matrix X, Matrix T, const size_t epochs, float learningRate)
{
	epoch += epochs;
	xMean = Matrix::mean(X, false);
	xStd = Matrix::standardDeviations(X, false);
	tMean = Matrix::mean(T, false);
	tStd = Matrix::standardDeviations(T, false);

	//Standardize
	X = (X - xMean) / xStd;
	T = (T - tMean) / tStd;
	//Train
	learningRate /= X.getDimensions().first * T.getDimensions().second;
	for (size_t i = 0; i < epochs; ++i)
	{
		Matrix Y = forward(X);
		std::vector<Matrix> grads = std::move(gradients(T));
		for (int j = 0; j < weights.size(); ++j)
		{
			weights[j] += learningRate * grads[(grads.size() - j) - 1];
		}

		error.push_back(rmse(T, Y));
	}
}

Matrix NeuralNetwork::use(Matrix X)
{
	if (epoch == 0)throw std::exception("Cannot use the Neural Network without training");
	X = (X - xMean) / xStd;
	Matrix Y = forward(X);
	return (Y * tStd) + tMean;
}

float NeuralNetwork::rmse(const Matrix& T, const Matrix& Y) const
{
	Matrix diff = (T - Y) * tStd;
	return std::sqrt(Matrix::mean(Matrix::square(diff)));
}

Matrix& NeuralNetwork::forward(const Matrix& X)
{
	Z[0] = X;
	for (size_t i = 0; i < weights.size() - 1; ++i)
	{
		Matrix result(std::move(Matrix::addOnes(Z[i]) * weights[i]));
		result.activateTanH();
		Z[i + 1] = std::move(result);//Would need to compare std move performance knowing a delete is involved..
	}
	Z.back() = std::move(Matrix::addOnes(Z[Z.size() - 2]) * weights.back());
	return Z.back();
}

std::vector<Matrix> NeuralNetwork::gradients(const Matrix& T) const
{
	//Reverse order for backpropagation (order of dependecies)
	std::vector<Matrix> grads(weights.size());
	//The reverse order is based on the weight count, and aligns with
	//	the indices of the Z matrices
	//	Notably, ||Z|| === ||W|| + 1
	//		but the first [0, ||W||) elements are aligned
	//	Therefore, can just reverse iterate over W's indices
	//		and reuse both for W[i] and Z[i]
	Matrix delta = std::move(T - Z.back());
	for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i)
	{
		Matrix dW = std::move(Matrix::transpose((Matrix::addOnes(Z[i]))) * delta);
		grads.emplace_back(std::move(dW));
		delta = std::move(Matrix::componentwise((delta * Matrix::transpose(weights[i], 1)),
			(1.0f - Matrix::square(Z[i]))));
	}
	return grads;
}

std::ostream& operator<<(std::ostream& out, const NeuralNetwork& mat)
{
	return out << mat.getInfo();
}