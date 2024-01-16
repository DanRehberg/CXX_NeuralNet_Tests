# CXX_NeuralNet_Tests
Testing performance of parallel matrix operations in a Neural Network

Groups of tests split by directories.
- __/UnitTests__
  - Contain the Matrix function and performance testing code
- __/NeuralNetworkTests__
  - Contain performance test between NeuralNetwork with and without parallel Matrix operations
- __/ThreadMemAtomicTests__
  - Contain changes to make atomic dot product case faster
  - Contains an altered **ThreadPool** class
    - N tasks per thread passed along in function, enabling local memory of work
    - Faster than adding a class (**ThreadMemory**) for memory (due to locality issues)
  
**Currently**, parallel matrix operations are limited to multiplication with three multiplication options currently available in the UnitTests workspace. **Additional tests added** in the ThreadMemAtomicTests workspace.

## Final Analysis Multiplication Performance
**Method**: __Using Ryzen 9 5900HS; average time of 50 and 100 iterations when multiplying two 80x80 matrices together. **Exception**, Case B is significantly slower, so only used 40x40 matrices.__

**Case A**: Dot product per thread
**Case B**: Contemporary __log base 2__ approach (batches/dispatches of threads per pair of terms to sum)
**Case C.1**: Original C variation, dot product terms atomically added to ensure thread safety in shared dot product
**Case C.2**: ThreadPool changed so threads can store local summation to reduce number of atomic operations
**Case C.3**: Same as C.2 but terms are integers instead of floats (testing int ops vs flops)
**Case C.4** Same as C.3, but replaced several integer divisions with conditional branching

| Method      | 50 Iterations Time (ms) | 100 iterations Time (ms)|
| :---        |    :----:   |     :----: |
| Serial      | 0.50       | 0.49 |
| Case A   | 0.02 | 0.03 |
| Case B (40x40)| 0.22 | 0.21 |
| Case C.1| 0.26| 0.27 |
| Case C.2| 0.08 | 0.08 |
| Case C.3| 0.04  | 0.05 |
| Case C.4| 0.1 | 0.1 |

## Discussion
Averages were consistent between 50 and 100 cycles, though first run of each application was discarded to ensure more than adequate __"warm starting"__ for reliable metrics (as seen in each __main__ several different performance tests are run in sequences; latter tests might benefit more than initial if not warm starting). The contemporary approach is better suited for lock step, as found in GPU implementations, and was ill fit to this CPU implementation. The simplest implementation was the fastest and is used, and will continue to be used, in the Neural Network class. The first atomic test required an atomic operation per term being summed, resulting in slow performance. Worse performance would be expected for matrices with few dot products but several working threads because they would more often share a common atomic integer to sum into. Subsequent atomic cases limited atomic operations by altering the ThreadPool used: _expected a function pointer which took in the start and end job indices to be managed by the worker thread rather than the ThreadPool class_. This enabled local memory for summing values until reaching the end of a dot product, and then performing an atomic operation. This got performance significantly closer to the best case found in **Case A**. This apporach performed better when always using integers rather than summing floats and scaling up before casting to an integer to perform the atomic addition. Case **C.4** was worse than **C.3** which seems reasonable as index changes are infrequent and conditional branching might not meet a threshold of tasks to perform in order to start showing greater performance over additional integer division operations (in both division operations and modulus).

## Pending
With the given parallel matrix multiplication method chosen, additional parallel functions will be added to further enhance performance with large neural network structures.
