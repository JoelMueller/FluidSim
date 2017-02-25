#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <builtin_types.h>

// undo defines from Windows.h
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{	
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}

inline float clamp(float x, float minX, float maxX)
{
	return std::max(minX, std::min(maxX, x));
}