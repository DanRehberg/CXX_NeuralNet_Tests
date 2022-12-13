/*
Author: Dan Rehberg
Modified: 12/8/2022
Purpose: Implementation of a neural network to test
	the performance of various matrix multiplication implementations.
The goal is to see how well atomic operations fit into machine
	learning -- fitting primarily into the concern of
	dependencies during the summation of dot prouct operations.
*/

#ifndef __Neural_Network_Parallel__
#define __Neural_Network_Parallel__

#include <ostream>
#include <vector>
#include <string>
#include "SerialMatrix.hpp"
#include "ThreadPool.hpp"

#define Matrix SerialMatrix

class NeuralNetworkParallel
{
private:
	NeuralNetworkParallel();
public:
	NeuralNetworkParallel(const NeuralNetworkParallel& cp) = delete;
	NeuralNetworkParallel(const size_t inputCount, const std::vector<size_t>& hiddenCount, const size_t outputCount);
	~NeuralNetworkParallel();

	void testWeights();

	std::string getInfo() const;
	void train(Matrix X, Matrix T, const size_t epochs, float learningRate);

	Matrix use(Matrix X);
private:
	NeuralNetworkParallel& operator=(const NeuralNetworkParallel& cp);

	float rmse(const Matrix& T, const Matrix& Y) const;
	Matrix& forward(const Matrix& X);
	std::vector<Matrix> gradients(const Matrix& T);

	size_t input, output, epoch;
	std::vector<size_t> hidden;
	std::vector<Matrix> weights, Z;
	std::vector<float> error;
	Matrix xMean, xStd, tMean, tStd;
	ThreadPool pool;

	static Matrix tempM;
};

std::ostream& operator<<(std::ostream& out, const NeuralNetworkParallel& mat);

#endif