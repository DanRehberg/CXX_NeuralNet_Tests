/*
Author: Dan Rehberg
Modified Date: 12/9/2022
*/
#include <iostream>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>
#include "SerialMatrix.hpp"
#include "ThreadPool.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkParallel.hpp"

int main()
{
	
	


	try
	{
		NeuralNetwork nn(1, { 3,2 }, 1);
		std::cout << nn << '\n';

		std::vector<std::vector<float>> xData;
		std::vector<std::vector<float>> tData;
		for (unsigned int i = 0; i < 10; ++i)
		{
			float val = static_cast<float>(i);
			float tar = std::sin(val) + 0.01f * (val * val);
			std::vector<float> rX;
			std::vector<float> rT;
			rX.push_back(val);
			rT.push_back(tar);
			xData.push_back(rX);
			tData.push_back(rT);
		}

		SerialMatrix X(xData);
		SerialMatrix T(tData);
		std::cout << X << "\n\n";
		std::cout << T << "\n";

		nn.testWeights();
		nn.train(X, T, 1, 0.1f);

		
	}
	catch (std::exception err)
	{
		std::cout << err.what() << "\n";
	}

	std::cout << "Network Case 2\n";
	try
	{
		NeuralNetwork nn(1, { 10,5 }, 1);
		std::cout << nn << '\n';

		std::vector<std::vector<float>> xData;
		std::vector<std::vector<float>> tData;
		for (unsigned int i = 0; i < 10; ++i)
		{
			float val = static_cast<float>(i);
			float tar = std::sin(val) + 0.01f * (val * val);
			std::vector<float> rX;
			std::vector<float> rT;
			rX.push_back(val);
			rT.push_back(tar);
			xData.push_back(rX);
			tData.push_back(rT);
		}

		SerialMatrix X(xData);
		SerialMatrix T(tData);
		std::cout << X << "\n\n";
		std::cout << T << "\n";

		nn.testWeights();

		unsigned int elapsedTime = 0;
		std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
		for (unsigned int i = 0; i < 50; ++i)
		{
			startTime = std::chrono::steady_clock::now();
			nn.train(X, T, 5000, 0.1f);
			endTime = std::chrono::steady_clock::now();
			elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		}
		std::cout << "total elapse and average: " << elapsedTime << " " <<
			(static_cast<float>(elapsedTime) / 50.0f) << "\n";

	}
	catch (std::exception err)
	{
		std::cout << err.what() << "\n";
	}

	std::cout << "Parallel Network Case 2\n";
	try
	{
		NeuralNetworkParallel nn(1, { 10,5 }, 1);
		std::cout << nn << '\n';

		std::vector<std::vector<float>> xData;
		std::vector<std::vector<float>> tData;
		for (unsigned int i = 0; i < 10; ++i)
		{
			float val = static_cast<float>(i);
			float tar = std::sin(val) + 0.01f * (val * val);
			std::vector<float> rX;
			std::vector<float> rT;
			rX.push_back(val);
			rT.push_back(tar);
			xData.push_back(rX);
			tData.push_back(rT);
		}

		SerialMatrix X(xData);
		SerialMatrix T(tData);
		std::cout << X << "\n\n";
		std::cout << T << "\n";

		nn.testWeights();

		unsigned int elapsedTime = 0;
		std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
		for (unsigned int i = 0; i < 50; ++i)
		{
			startTime = std::chrono::steady_clock::now();
			nn.train(X, T, 5000, 0.1f);
			endTime = std::chrono::steady_clock::now();
			elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		}
		std::cout << "parallel total elapse and average: " << elapsedTime << " " <<
			(static_cast<float>(elapsedTime) / 50.0f) << "\n";

	}
	catch (std::exception err)
	{
		std::cout << err.what() << "\n";
	}

	std::cout << "Network Case 3 Big Matrices\n";
	try
	{
		NeuralNetwork nn(1, { 100,50 }, 1);
		std::cout << nn << '\n';

		std::vector<std::vector<float>> xData;
		std::vector<std::vector<float>> tData;
		for (unsigned int i = 0; i < 10; ++i)
		{
			float val = static_cast<float>(i);
			float tar = std::sin(val) + 0.01f * (val * val);
			std::vector<float> rX;
			std::vector<float> rT;
			rX.push_back(val);
			rT.push_back(tar);
			xData.push_back(rX);
			tData.push_back(rT);
		}

		SerialMatrix X(xData);
		SerialMatrix T(tData);
		std::cout << X << "\n\n";
		std::cout << T << "\n";

		nn.testWeights();

		unsigned int elapsedTime = 0;
		std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
		for (unsigned int i = 0; i < 50; ++i)
		{
			startTime = std::chrono::steady_clock::now();
			nn.train(X, T, 5000, 0.1f);
			endTime = std::chrono::steady_clock::now();
			elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		}
		std::cout << "total elapse and average: " << elapsedTime << " " <<
			(static_cast<float>(elapsedTime) / 50.0f) << "\n";

	}
	catch (std::exception err)
	{
		std::cout << err.what() << "\n";
	}

	std::cout << "Parallel Network Case 3 Big Matrices\n";
	try
	{
		NeuralNetworkParallel nn(1, { 100,50 }, 1);
		std::cout << nn << '\n';

		std::vector<std::vector<float>> xData;
		std::vector<std::vector<float>> tData;
		for (unsigned int i = 0; i < 10; ++i)
		{
			float val = static_cast<float>(i);
			float tar = std::sin(val) + 0.01f * (val * val);
			std::vector<float> rX;
			std::vector<float> rT;
			rX.push_back(val);
			rT.push_back(tar);
			xData.push_back(rX);
			tData.push_back(rT);
		}

		SerialMatrix X(xData);
		SerialMatrix T(tData);
		std::cout << X << "\n\n";
		std::cout << T << "\n";

		nn.testWeights();

		unsigned int elapsedTime = 0;
		std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
		for (unsigned int i = 0; i < 50; ++i)
		{
			startTime = std::chrono::steady_clock::now();
			nn.train(X, T, 5000, 0.1f);
			endTime = std::chrono::steady_clock::now();
			elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		}
		std::cout << "parallel total elapse and average: " << elapsedTime << " " <<
			(static_cast<float>(elapsedTime) / 50.0f) << "\n";

	}
	catch (std::exception err)
	{
		std::cout << err.what() << "\n";
	}

	return 0;
}