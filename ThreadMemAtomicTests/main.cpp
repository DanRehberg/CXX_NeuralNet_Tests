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
	//Warm start with initial pass..

	std::cout << "\nMatrix Parallel Int performance";
	ThreadPool pool(std::thread::hardware_concurrency() - 1);
	std::cout << "\ninsert an integer for the number of tests...\n";
	unsigned int trials = 0;
	std::cin >> trials;
	try
	{
		std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
		
		std::vector<float> rows;
		//Build upto a 20x20 matrix to see when serial slows down
		for (unsigned int i = 0; i < 80; ++i)
		{
			rows.push_back(static_cast<float>(i + 1));
			std::vector<std::vector<float>> vectorMat;
			for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
			Mat testing(vectorMat);
			Mat A = testing;// ({ {-1.f,2.f},{3.f,2.f},{2.f,3.f} });
			Mat B = testing;// ({ {4.f,5.f,6.f},{6.f,-5.f,4.f} });
			//Note, both the parallel matrix and the serial matrix have memory allocated outside of performance testing.
			//	This ensures we minimize the time delta to represent the logic to solve the problem, not the setup work.
			//	The setup work should in fact be near identical as they both merely instantiated a ParallelMatrix object of certin dimensions.
			Mat::setParallelMatrixOps(A, B, true);
			Mat C(A.getDimensions().first, B.getDimensions().second);
			unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
			unsigned int totalSums = tempSize * A.getDimensions().second;
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
			{
				//Mat::setParallelMatrixOps(A, B, true); //Technically this is needed to generate the
				//	correct (within epsilon of err) result because it resets the atomic values
				//pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialC0));
				//pool.dispatch(totalSums, &(Mat::parallelTrialC0Alt));
				pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAlt));
				//pool.dispatch(tempSize, &(Mat::parallelTrialC1));
			}
			endTime = std::chrono::steady_clock::now();
			unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
				C = A * B;
			endTime = std::chrono::steady_clock::now();
			//std::cout << "Same as serial result: " << ((C == Mat::getParallelResult()) ? "true" : "false") << '\n';
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
		//Build upto a 20x20 matrix to see when serial slows down
		for (unsigned int i = 0; i < 80; ++i)
		{
			rows.push_back(static_cast<float>(i + 1));
			std::vector<std::vector<float>> vectorMat;
			for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
			MatF testing(vectorMat);
			MatF A = testing;// ({ {-1.f,2.f},{3.f,2.f},{2.f,3.f} });
			MatF B = testing;// ({ {4.f,5.f,6.f},{6.f,-5.f,4.f} });
			//Note, both the parallel matrix and the serial matrix have memory allocated outside of performance testing.
			//	This ensures we minimize the time delta to represent the logic to solve the problem, not the setup work.
			//	The setup work should in fact be near identical as they both merely instantiated a ParallelMatrix object of certin dimensions.
			MatF::setParallelMatrixOps(A, B, true);
			MatF C(A.getDimensions().first, B.getDimensions().second);
			unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
			unsigned int totalSums = tempSize * A.getDimensions().second;
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
			{
				//Mat::setParallelMatrixOps(A, B, true); //Technically this is needed to generate the
				//	correct (within epsilon of err) result because it resets the atomic values
				//pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialC0));
				//pool.dispatch(totalSums, &(Mat::parallelTrialC0Alt));
				pool.dispatch(totalSums, &(MatF::parallelTrialC0AltAlt));
				//pool.dispatch(tempSize, &(Mat::parallelTrialC1));
			}
			endTime = std::chrono::steady_clock::now();
			unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
				C = A * B;
			endTime = std::chrono::steady_clock::now();
			//std::cout << "Same as serial result: " << ((C == Mat::getParallelResult()) ? "true" : "false") << '\n';
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
		//Build upto a 20x20 matrix to see when serial slows down
		for (unsigned int i = 0; i < 80; ++i)
		{
			rows.push_back(static_cast<float>(i + 1));
			std::vector<std::vector<float>> vectorMat;
			for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
			Mat testing(vectorMat);
			Mat A = testing;// ({ {-1.f,2.f},{3.f,2.f},{2.f,3.f} });
			Mat B = testing;// ({ {4.f,5.f,6.f},{6.f,-5.f,4.f} });
			//Note, both the parallel matrix and the serial matrix have memory allocated outside of performance testing.
			//	This ensures we minimize the time delta to represent the logic to solve the problem, not the setup work.
			//	The setup work should in fact be near identical as they both merely instantiated a ParallelMatrix object of certin dimensions.
			Mat::setParallelMatrixOps(A, B, true);
			Mat C(A.getDimensions().first, B.getDimensions().second);
			unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
			unsigned int totalSums = tempSize * A.getDimensions().second;
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
			{
				//Mat::setParallelMatrixOps(A, B, true); //Technically this is needed to generate the
				//	correct (within epsilon of err) result because it resets the atomic values
				//pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialC0));
				//pool.dispatch(totalSums, &(Mat::parallelTrialC0Alt));
				pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAlt));
				//pool.dispatch(tempSize, &(Mat::parallelTrialC1));
			}
			endTime = std::chrono::steady_clock::now();
			unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			startTime = std::chrono::steady_clock::now();
			for (unsigned int i = 0; i < trials; ++i)
				C = A * B;
			endTime = std::chrono::steady_clock::now();
			//std::cout << "Same as serial result: " << ((C == Mat::getParallelResult()) ? "true" : "false") << '\n';
			unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			unsigned int scale = testing.getDimensions().first;
			std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
				timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
		}
	}
	catch (...)
	{
	}
	char wait = 'n';
	std::cin >> wait;

	return 0;
}
