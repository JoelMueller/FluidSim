#pragma once

#include <Matog.h>
#include "matog_gen/Array2D.h"

namespace FluidSim
{
	struct cudaInfo
	{
		CUdevice   device;
		CUcontext  context;
		CUmodule   module;
		CUfunction advection_function;
		CUfunction jacobi_function;
		CUfunction divergence_function;
		CUfunction subtractGradient_function;
		CUfunction boundary_function;
		CUfunction addInk_function;
		CUfunction convertToColor_function;
		CUfunction convertToColor2_function;
		size_t     totalGlobalMem;

		int height, width;
        int threads_x, threads_y;
	};

	void advect(cudaInfo & info, Array2D::Device *q, Array2D::Device *qNew, Array2D::Device *u, Array2D::Device *v, float dt, float rdx);
	void jacobi(cudaInfo & info, Array2D::Device *x, Array2D::Device *xNew, Array2D::Device *b, float alpha, float rbeta);
	void divergence(cudaInfo & info, Array2D::Device *u, Array2D::Device *v, Array2D::Device *div, float halfrdx);
	void subtractGradient(cudaInfo & info, Array2D::Device *p, Array2D::Device *u, Array2D::Device *v, Array2D::Device *uNew, Array2D::Device *vNew, float halfrdx);
	void boundary(cudaInfo & info, Array2D::Device *x, float scale);
	void addInk(cudaInfo & info, Array2D::Device *u, Array2D::Device *v, Array2D::Device *ink, int x, int y, float u_, float v_, float ink_);
	void convertToColor(cudaInfo & info, CUdeviceptr color, Array2D::Device *x);
	void convertToColor2(cudaInfo & info, CUdeviceptr color, Array2D::Device *r, Array2D::Device *g, Array2D::Device *b);
};