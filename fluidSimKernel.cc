#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "fluidSimKernel.h"
#include "common.h"

namespace FluidSim
{
	void advect(cudaInfo & info, Array2D::Device *q, Array2D::Device *qNew, Array2D::Device *u, Array2D::Device *v, float dt, float rdx)
	{
		void *args[7] = { q, qNew, u, v, &dt, &rdx, 0 };

		CHECK(cuLaunchKernel(info.advection_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void jacobi(cudaInfo & info, Array2D::Device *x, Array2D::Device *xNew, Array2D::Device *b, float alpha, float rbeta)
	{
		void *args[6] = { x, xNew, b, &alpha, &rbeta, 0 };

		CHECK(cuLaunchKernel(info.jacobi_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void divergence(cudaInfo & info, Array2D::Device *u, Array2D::Device *v, Array2D::Device *div, float halfrdx)
	{
		void *args[5] = { u, v, div, &halfrdx, 0 };

		CHECK(cuLaunchKernel(info.divergence_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void subtractGradient(cudaInfo & info, Array2D::Device *p, Array2D::Device *u, Array2D::Device *v, Array2D::Device *uNew, Array2D::Device *vNew, float halfrdx)
	{
		void *args[7] = { p, u, v, uNew, vNew, &halfrdx, 0 };

		CHECK(cuLaunchKernel(info.subtractGradient_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void boundary(cudaInfo & info, Array2D::Device *x, float scale)
	{
		void *args[3] = { x, &scale, 0 };

		CHECK(cuLaunchKernel(info.boundary_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void addInk(cudaInfo & info, Array2D::Device *u, Array2D::Device *v, Array2D::Device *ink, int x, int y, float u_, float v_, float ink_)
	{
		void *args[9] = { u, v, ink, &x, &y, &u_, &v_, &ink_, 0 };

		CHECK(cuLaunchKernel(info.addInk_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void convertToColor(cudaInfo & info, CUdeviceptr color, Array2D::Device *x)
	{
		void *args[3] = { &color, &x, 0 };

		CHECK(cuLaunchKernel(info.convertToColor_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}

	void convertToColor2(cudaInfo & info, CUdeviceptr color, Array2D::Device *r, Array2D::Device *g, Array2D::Device *b)
	{
		void *args[5] = { &color, &r, &g, &b, 0 };

		CHECK(cuLaunchKernel(info.convertToColor2_function, div_up(info.width, info.threads_x), div_up(info.height, info.threads_y), 1,
			info.threads_x, info.threads_y, 1,
			0, 0, args, 0));
	}
}