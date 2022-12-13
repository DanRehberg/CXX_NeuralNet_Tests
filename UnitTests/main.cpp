/*
Author: Dan Rehberg
Modified Date: 12/9/2022
*/
#include <iostream>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>
#include "ParallelMatrix.hpp"
#include "ThreadPool.hpp"

#define Mat ParallelMatrix

int main()
{
	//Test cases for Zero sized matrices
	std::cout << "\nZero matrices test cases\n";
	try
	{
		Mat test = Mat(0, 0);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	try
	{
		Mat test(0, 0);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	//Test cases for Types of object creation
	std::cout << "\nConstructor test cases\n";
	try
	{
		Mat test = Mat(1, 1);
		std::cout << "Copy Constructor invocation: " << test.getDimensions().first << " " << test.getDimensions().second << '\n';
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	try
	{
		Mat test(1, 1);
		std::cout << "Primary Constructor invocation: " << test.getDimensions().first << " " << test.getDimensions().second << "\n";
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	std::cout << "\nMatrix copy and = test cases\n";
	try
	{
		Mat testA(4, 4);
		std::cout << "EXPECT: New Data\n";
		Mat testB = testA;
		Mat testC(16, 1);
		std::cout << "EXPECT: Reuse Data\n";
		testC = testA;
		Mat testD(5, 2);
		std::cout << "EXPECT: New Data\n";
		testA = testD;
	}
	catch (...)
	{
	}

	std::cout << "\nMatrix vector initializer list good and bad\n";
	try
	{
		Mat testA({ {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f} });
		Mat testB({ {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f} });
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	try
	{
		std::vector<std::vector<float>> tempV = { {} };
		Mat testA(tempV);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}
	try
	{
		std::vector<std::vector<float>> tempV = { {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {2.0f, 1.0f} };
		Mat testA(1, 1);
		testA = tempV;
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}
	try
	{
		std::vector<std::vector<float>> tempV = { {} };
		Mat testA(1, 1);
		testA = tempV;
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	std::cout << "\nMatrix array failure\n";
	try
	{
		std::array<std::array<float, 0>, 1> tempA;
		Mat testA(tempA);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}
	try
	{
		std::array<std::array<float, 0>, 1> tempA;
		Mat testA(1, 1);
		testA = tempA;
	}
	catch (std::exception err)
	{
		std::cout << err.what() << '\n';
	}

	std::cout << "\nMatrix container copy and = with vector or array\n";
	try
	{
		std::vector<std::vector<float>> tempV = { {2.0f, 1.0f} };
		std::array<std::array<float, 2>, 1> tempA = { {-3.0f, -2.0f} };
		std::cout << "EXPECT: New Data\n";
		Mat testA(tempV);
		std::cout << "EXPECT: New Data\n";
		Mat testB(tempA);
		Mat testC(1, 1);
		std::cout << "EXPECT: New Data\n";
		testC = tempV;
		Mat testD(5, 50);
		std::cout << "EXPECT: New Data\n";
		testD = tempA;
		std::cout << "EXPECT: Reuse Data\n";
		testD = tempV;
		std::cout << "EXPECT: Reuse Data\n";
		testD = tempA;
	}
	catch (...)
	{

	}
	//tried here
	std::cout << "\nMatrix values stored\n";
	try
	{
		Mat testA(5, 5);
		std::cout << "testA: " << testA << '\n';
		Mat testB({ {1.0f, 2.0f, 3.0f} });
		std::cout << "testB: " << testB << '\n';
		Mat testC({ {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f}, {-3.0f, -2.0f, -1.0f} });
		std::cout << "testC: " << testC << '\n';
	}
	catch (...)
	{
	}

	std::cout << "\nMatrix multiplication\n";
	try
	{
		Mat A({ {-1.f,2.f},{3.f,2.f},{2.f,3.f} });
		Mat B({ {4.f,5.f,6.f},{6.f,-5.f,4.f} });
		Mat C = A * B;
		std::cout << C << "\n";
	}
	catch (...)
	{
	}
	{
		ThreadPool pool(std::thread::hardware_concurrency() - 1);

		std::cout << "\nMatrix parallel case A multiplication\n";
		try
		{
			Mat A({ {-1.f,2.f},{3.f,2.f},{2.f,3.f} });
			Mat B({ {4.f,5.f,6.f},{6.f,-5.f,4.f} });
			Mat::setParallelMatrixOps(A, B, true);
			unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
			std::cout << "pool begin\n";
			pool.dispatch(tempSize, &(Mat::parallelTrialA));
			std::cout << "pool end\n";
			Mat C = Mat::getParallelResult();
			Mat C1 = A * B;
			std::cout << "Same as serial result: " << ((C == C1) ? "true" : "false") << '\n';
		}
		catch (...)
		{
		}


		std::cout << "\nMatrix Parallel (A) performance comparison";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::cout << "\ninsert an integer for the number of tests...\n";
			unsigned int trials = 0;
			std::cin >> trials;
			std::vector<float> rows;
			for (unsigned int i = 0; i < 80; ++i)
			{
				rows.push_back(static_cast<float>(i + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
				Mat testing(vectorMat);
				Mat A = testing;
				Mat B = testing;
				//Note, both the parallel matrix and the serial matrix have memory allocated outside of performance testing.
				//	This ensures we minimize the time delta to represent the logic to solve the problem, not the setup work.
				//	The setup work should in fact be near identical as they both merely instantiated a ParallelMatrix object of certin dimensions.
				Mat::setParallelMatrixOps(A, B, true);
				Mat C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					pool.dispatch(tempSize, &(Mat::parallelTrialA));
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



#if DEBUG_MATRIX == 2
		std::cout << "\nVerify total number of products are occurring in first stage of logbase 2 method\n";
		try
		{
			Mat A({ {1.f,2.f},{3.f,4.f},{5.f,6.f} });
			Mat B({ {7.f,8.f,9.f},{10.f,11.f,12.f} });
			//Mat A({ {1,2,3}, {4,5,6} });
			//Mat B({ {7,8}, {9,10}, {11, 12} });
			unsigned int cSize = A.getDimensions().first * B.getDimensions().second;
			Mat::setParallelMatrixOps(A, B, true);
			pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialB0));
			std::cout << "Total multiplications: " << Mat::multiplications.load(std::memory_order_relaxed) << "\n";
			std::cout << "mD: " << Mat::getMD() << "\n";
			Mat C = A * B;
			std::cout << "\nC:" << C << "\n";
			unsigned int N = A.getDimensions().second;// cSize;
			std::cout << "size: " << N << "\n";
			Mat::setTrialB1(N);
			N = (N + 1) >> 1;
			std::cout << "size: " << N << "\n";
			while (N > 1)
			{
				std::cout << "pre dispatch, thread count: " << N << "\n";
				pool.dispatch(N, &(Mat::parallelTrialB1));
				Mat::setTrialB1(N);
				N = (N + 1) >> 1;
			}
			std::cout << "N: " << N << "\n";
			//Convert the final dispatch below to the last summation and setting
			//	the result -- easily parallel by the number of total summations that occurred.
			pool.dispatch(N, &(Mat::parallelTrialB1));
			std::cout << "post mD: " << Mat::getMD() << "\n\n";
			pool.dispatch(cSize, &(Mat::parallelTrialB2));
			std::cout << Mat::getParallelResult() << "\n";
		}
		catch (...)
		{
		}
#endif



		std::cout << "\nAtomic Parallel Matrix Case\n";
		try
		{
			Mat A({ {1.f,2.f},{3.f,4.f},{5.f,6.f} });
			Mat B({ {7.f,8.f,9.f},{10.f,11.f,12.f} });
			unsigned int cSize = A.getDimensions().first * B.getDimensions().second;
			Mat::setParallelMatrixOps(A, B, true);
			pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialC0));
			pool.dispatch(cSize, &(Mat::parallelTrialC1));
			std::cout << "Result: " << Mat::getParallelResult() << "\n";

		}
		catch (...)
		{
		}

		//Performance testing the Log base 2 approach
		std::cout << "\nMatrix Parallel (B) performance comparison";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::cout << "\ninsert an integer for the number of tests...\n";
			unsigned int trials = 0;
			std::cin >> trials;
			std::vector<float> rows;
			for (unsigned int i = 0; i < 40; ++i)
			{
				rows.push_back(static_cast<float>(i + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
				Mat testing(vectorMat);
				Mat A = testing;
				Mat B = testing;
				Mat::setParallelMatrixOps(A, B, true);
				Mat C(A.getDimensions().first, B.getDimensions().second);
				unsigned int tempSize = A.getDimensions().first * B.getDimensions().second;
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
				{
					pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialB0));
					unsigned int N = A.getDimensions().second;
					Mat::setTrialB1(N, true);
					N = (N + 1) >> 1;
					while (N > 1)
					{
						pool.dispatch(N, &(Mat::parallelTrialB1));
						Mat::setTrialB1(N);
						N = (N + 1) >> 1;
					}
					//Convert the final dispatch below to the last summation and setting
					//	the result -- easily parallel by the number of total summations that occurred.
					pool.dispatch(N, &(Mat::parallelTrialB1));
					pool.dispatch(tempSize, &(Mat::parallelTrialB2));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				std::cout << "Same as serial result: " << ((C == Mat::getParallelResult()) ? "true" : "false") << '\n';
				unsigned int timeB = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				unsigned int scale = testing.getDimensions().first;
				std::cout << "Matrix Multiplication of: " << scale << "x" << scale << "; Parallel time: " << timeA << " (" << (static_cast<float>(timeA) / static_cast<float>(trials)) << ")" << " ms; Serial time: " <<
					timeB << " (" << (static_cast<float>(timeB) / static_cast<float>(trials)) << ")" << " ms\n";
			}
		}
		catch (...)
		{
		}

		std::cout << "\nMatrix Parallel (C) performance comparison";
		try
		{
			std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
			std::cout << "\ninsert an integer for the number of tests...\n";
			unsigned int trials = 0;
			std::cin >> trials;
			std::vector<float> rows;
			for (unsigned int i = 0; i < 80; ++i)
			{
				rows.push_back(static_cast<float>(i + 1));
				std::vector<std::vector<float>> vectorMat;
				for (unsigned int j = 0; j <= i; ++j) vectorMat.push_back(rows);
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
					//Mat::setParallelMatrixOps(A, B, true); //Technically this is needed to generate the
					//	correct (within epsilon of err) result because it resets the atomic values
					//pool.dispatch(A.getDimensions().second, &(Mat::parallelTrialC0));
					//pool.dispatch(totalSums, &(Mat::parallelTrialC0Alt));
					//Just testing the multiplication operation and not conversion back to a float
					pool.dispatch(totalSums, &(Mat::parallelTrialC0AltAlt));
					//pool.dispatch(tempSize, &(Mat::parallelTrialC1));
				}
				endTime = std::chrono::steady_clock::now();
				unsigned int timeA = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
				startTime = std::chrono::steady_clock::now();
				for (unsigned int i = 0; i < trials; ++i)
					C = A * B;
				endTime = std::chrono::steady_clock::now();
				//Will fluctuate between equivalence and not due to int->float changes
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
	}


	return 0;
}