//
// CSU33014 Lab 1
//

// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>

#include "lab1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void lab1_routine0(float * restrict a, float * restrict b,
		    float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void lab1_vectorized0(float * restrict a, float * restrict b,
		    float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float lab1_routine1(float * restrict a, float * restrict b,
		     int size) {
  float sum = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float lab1_vectorized1(float * restrict a, float * restrict b,
		     int size) {
  //We will eventually return a float, so we can initialize this variable sum.
  float sum = 0.0;
  //We want to vectorize the loop, hence we can take 4 iterations at a time.
  //If the number of iterations (i.e sum) is odd, then we will have some remaining iterations at the end after doing 4 at a time.
  //These iterations can be done with non vectorized code at the end!
  int remainder = size%4;
  float sum_arr[4];
  int i;
  for(i=0; i<size-remainder; i=i+4){
    //a4 = {a[i], a[i+1], a[i+2], a[i+3]}
    __m128 a4 = _mm_loadu_ps(&a[i]);

    //b4 = {b[i], b[i+1], b[i+2], b[i+3]}
    __m128 b4 = _mm_loadu_ps(&b[i]);

    //a4 = {a[i], a[i+1], a[i+2], a[i+3]} * {b[i], b[i+1], b[i+2], b[i+3]}
    a4 = _mm_mul_ps(a4, b4);

    //Calculations that occur:
    //sum_arr[0]=a[i]*b[i];
    //sum_arr[1]=a[i+1]*b[i+1];
    //sum_arr[2]=a[i+2]*b[i+2];
    //sum_arr[3]=a[i+3]*b[i+3];
    _mm_storeu_ps(sum_arr, a4);

    //Since a[i] * b[i] is now computed for 4 iterations, now we need to add sum to each.
    sum = sum + sum_arr[0];
    sum = sum + sum_arr[1];
    sum = sum + sum_arr[2];
    sum = sum + sum_arr[3];
  }

  //Any remaining iterations that we can do in a non vectorized manner.
  for(;i<size;i++){
    sum = sum + a[i] * b[i];
  }
  return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void lab1_routine2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

// in the following, size can have any positive value
void lab1_vectorized2(float * restrict a, float * restrict b, int size) {
  //We want to vectorize the loop, hence we can take 4 iterations at a time.
  //If the number of iterations (i.e sum) is odd, then we will have some remaining iterations at the end after doing 4 at a time.
  //These iterations can be done with non vectorized code at the end!
  int remainder = size%4;
  int i;
  //Mask of all 1's.
  __m128 allOnes = _mm_set1_ps((float)1);
  for(i=0; i<size-remainder; i=i+4){
    //b4 = {b[i], b[i+1], b[i+2], b[i+3]}
    __m128 b4 = _mm_loadu_ps(&b[i]);
    b4 = _mm_add_ps(b4, allOnes); //calculate denominator
    b4 = _mm_div_ps(allOnes, b4); //calculate operand 2, could also use b4 = _mm_rcp_ps(b4); to get the reciprocal
    b4 = _mm_sub_ps(allOnes, b4); //calculate final result by subtracting 1 from all 4 values of operand 2.
    _mm_storeu_ps(&a[i], b4);     //store result of iterations into a[i]
  }

  //Any remaining iterations that we can do in a non vectorized manner.
  for(;i<size;i++){
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void lab1_routine3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void lab1_vectorized3(float * restrict a, float * restrict b, int size) {
  // replace the following code with vectorized code
  //if a is less than 0, then a[i] = b[i], if a is greater than 0, then a[i] is the same 
  __m128 allZeros = _mm_setzero_ps();
  int remainder = size%4;
  int i;
  for(i=0; i<size-remainder; i=i+4){
    //{a[i], a[i+1], a[i+2], a[i+3]}
    __m128 a4 = _mm_loadu_ps(&a[i]);

    //Need to compute {a[i] < 0, a[i+1] < 0, a[i+2] < 0, a[i+3] < 0}
    //true=0xFFFFFFFF, false=0x0
    __m128 compare = _mm_cmplt_ps(a4, allZeros);

    //{b[v], b[v+1], b[v+2], b[v+3]}
    __m128 b4 = _mm_loadu_ps(&b[i]);

    //AND operator on b_vec and cmp_vec
    //Values remain where a[i+x] < 0
    //Example: a=[1,-1,3,4,-1], b=[5,1,2,1,6], compare=[0x0,0xFFFFFFFF,0x0,0x0,0xFFFFFFFF]
    //Hence compare && b = [0x0,1,0x0,0x0,6] i.e values only where a[i] < 0.
    b4 = _mm_and_ps(b4, compare);

    //NOT operator on compare
    //AND operator on a4 and compare
    //Values remain where a[i+x] >= 0
    //Example from before: a=[1,-1,3,4,-1], b=[5,1,2,1,6], compare=[0x0,0xFFFFFFFF,0x0,0x0,0xFFFFFFFF]
    //!compare = [0xFFFFFFFF,0x0,0xFFFFFFFF,0xFFFFFFFF,0x0]
    //Hence a4 && !compare = [1,0x0,3,4,0x0] i.e. values only when a[i] >= 0
    a4 = _mm_andnot_ps(compare, a4);

    //OR operator on a4 and b4
    //Store result into a[i]
    //Example from before: a4 = [1,0x0,3,4,0x0], b4 = [0x0,1,0x0,0x0,6]
    //Hence a4 || b4 = [1,1,3,4,6] i.e result of a[i] where all values > 0 have been swapped with b[i].
    _mm_storeu_ps(&a[i], _mm_or_ps(a4, b4));
  }

  //Any remaining iterations that we can do in a non vectorized manner.
  for (int i = 0; i < size; i++) {
    if (a[i] < 0.0) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float * restrict a, float * restrict b,
		       float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i] - b[i+1]*c[i+1];
    a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
  }
}

#define MASKFORSHUFFLE(a, b, c, d) ((d << 6) | (c << 4) | (b << 2) | (a << 0))
void lab1_vectorized4(float * restrict a, float * restrict b,
		       float * restrict  c) {
  // replace the following code with vectorized code
for (int i = 0; i < 2048; i += 4) {
    //load {b[i], b[i+1], b[i+2], b[i+3]}
    __m128 b4 = _mm_loadu_ps(&b[i]);
    //load {c[i], c[i+1], c[i+2], c[i+3]}
    __m128 c4 = _mm_loadu_ps(&c[i]);

    //{ b[i], b[i], b[i+2], b[i+2]}
    __m128 chooseB = _mm_shuffle_ps(b4, b4, MASKFORSHUFFLE(0, 0, 2, 2));

    //{ b[i]+c[i], b[i]*c[i+1], b[i+2]*c[i+2], b[i+2]*c[i+3}
    __m128 completedI = _mm_mul_ps(chooseB, c4);

    //{ c[i+1], c[i], c[i+3], c[i+2] }
    __m128 chooseC = _mm_shuffle_ps(c4, c4, MASKFORSHUFFLE(1, 0, 3, 2));

    //{ b[i+1], b[i+1], b[i+3], b[i+3]}
    chooseB = _mm_shuffle_ps(b4, b4, MASKFORSHUFFLE(1, 1, 3, 3));

    //{ c[i+1]*b[i+1], c[i]+b[i+1], c[i+3]*b[i+3], c[i+2]*b[i+3] }
    __m128 completedIPlus = _mm_mul_ps(chooseC, chooseB);

    //{( b[i]*c[i] )-( b[i+1]*c[i+1] ), ( b[i+1]*c[i] )+( b[i]*c[i+1] )
    //( b[i+2]*c[i+2] )-( b[i+3]*c[i+3] ), ( b[i+3]*c[i+2] )+( b[i+2]*c[i+3]
    completedI = _mm_addsub_ps(completedI, completedIPlus);

    //Store the result to a4
    _mm_storeu_ps(&a[i], completedI);
   }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void lab1_routine5(unsigned char * restrict a,
		    unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = b[i];
  }
}

void lab1_vectorized5(unsigned char * restrict a,
		       unsigned char * restrict b, int size) {
  // replace the following code with vectorized code
  int i;
  for (i = 0; i < size - 15; i += 16) {
    // load 16 chars
    __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]);
    // store 16 chars
    _mm_storeu_si128((__m128i *)&a[i], b16);
  }

  // parse sets of 4
  int *a_int = (int *)a;
  int *b_int = (int *)b;
  for (int z = i / 4; z < (size - 3) / 4; z++, i += 4) {
    a_int[z] = b_int[z];
  }

  // remaining multiples of 1
  for (; i < size; i++) {
    a[i] = b[i];
  }
}

/********************* routine 6 ***********************/

void lab1_routine6(float * restrict a, float * restrict b,
		       float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void lab1_vectorized6(float * restrict a, float * restrict b,
		       float * restrict c) {
    //replace the following code with vectorized code
  a[0] = 0.0;
  float notFinishedSum[4];
    //Constant value -> { c[0], c[1], c[2], c[3]}
  __m128 c4 = _mm_loadu_ps(c);
  for (int i = 1; i < 1023; i++) {
    //{b[i-1],b[i],b[i+1],b[i+2]}
    __m128 b4 = _mm_loadu_ps(&b[i - 1]);
    //{b[i-1]*c[0],b[i]*c[1],b[i+1]*c[2],b[i+2]*c[3]}
    b4 = _mm_mul_ps(b4, c4);
    //store result into a[i]
    _mm_storeu_ps(notFinishedSum, b4);
    //prevent floating point errors
    a[i] = notFinishedSum[0];
    a[i] += notFinishedSum[1];
    a[i] += notFinishedSum[2];
  }
  a[1023] = 0.0;  
}



