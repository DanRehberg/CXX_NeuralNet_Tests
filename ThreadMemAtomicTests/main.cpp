#include <iostream>
#include <chrono>
#include "ThreadMemory.hpp"
#include "ThreadPool.hpp"
#include "AtomicMatrix.hpp"
#include "IntAtomicMatrix.hpp"

#define Mat IntegerMatrix
#define MatF FloatMatrix

int main()
{
	{
		//Warm start with initial pass..

		std::cout << "\nWarming up with: Matrix Parallel Int performance";
		ThreadPool pool(std::thread::hardware_concurrency() - 1);
		std::cout << "\ninsert an integer for the number of tests...\n";
		unsigned int trials = 0;
		std::cin >> trials;
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;

			std::vector<float> rows;
			for (unsigned int m = 0; m < 80; ++m)
			{
				rows.push_back(static_cast<float>(m + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= m; ++j) vectorMat.push_back(rows);
				Mat testing(vectorMat);
				Mat A = testing;
				Mat B = testing;
				Mat::setParallelMatrixOps(A, B, true);
				Mat C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				unsigned int totalSums = tempSize * A.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
				{
					pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAlt));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				unsigned int scale = testing.getDimensions().first;
				std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
					timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
			}
		}
		catch (...)
		{
		}

		std::cout << "\nMatrix Parallel Float performance";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::vector<float> rows;
			for (unsigned int m = 0; m < 80; ++m)
			{
				rows.push_back(static_cast<float>(m + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= m; ++j) vectorMat.push_back(rows);
				MatF testing(vectorMat);
				MatF A = testing;
				MatF B = testing;
				MatF::setParallelMatrixOps(A, B, true);
				MatF C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				unsigned int totalSums = tempSize * A.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
				{
					pool.dispatch(totalSums, &(MatF::parallelTrialC0AltAlt));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				unsigned int scale = testing.getDimensions().first;
				std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
					timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
			}
		}
		catch (...)
		{
		}

		std::cout << "\nMatrix Parallel Int (B) performance";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::vector<float> rows;
			for (unsigned int m = 0; m < 80; ++m)
			{
				rows.push_back(static_cast<float>(m + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= m; ++j) vectorMat.push_back(rows);
				Mat testing(vectorMat);
				Mat A = testing;
				Mat B = testing;
				Mat::setParallelMatrixOps(A, B, true);
				Mat C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				unsigned int totalSums = tempSize * A.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
				{
					pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAlt));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				unsigned int scale = testing.getDimensions().first;
				std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
					timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
			}
		}
		catch (...)
		{
		}

		std::cout << "\nMatrix Parallel Int Alternative performance";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::vector<float> rows;
			for (unsigned int m = 0; m < 80; ++m)
			{
				rows.push_back(static_cast<float>(m + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= m; ++j) vectorMat.push_back(rows);
				Mat testing(vectorMat);
				Mat A = testing;
				Mat B = testing;
				Mat::setParallelMatrixOps(A, B, true);
				Mat C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				unsigned int totalSums = tempSize * A.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
				{
					pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAltAlt));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				unsigned int scale = testing.getDimensions().first;
				std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
					timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
			}
		}
		catch (...)
		{
		}
	}

	char wait = 'n';
	std::cin >> wait;

	return 0;
}
