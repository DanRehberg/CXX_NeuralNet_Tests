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
  
**Currently**, parallel matrix operations are limited to multiplication with three multiplication options currently available in the UnitTests workspace.
