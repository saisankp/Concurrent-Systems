# CSU33014 Lab 2: Parallel Multichannel Multikernel Convolution

## How to compile the code
```
gcc -O3 -msse4 conv-harness.c -fopenmp
```

## How to run the code with particular inputs
```
./a.out 256 256 7 128 128
```