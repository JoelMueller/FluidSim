#include "stdint.h"
#include "Array2D.cu"

extern __device__ float clamp(float x, float minX, float maxX){
	return max(minX, min(maxX, x));
}
extern __device__ int clamp(int x, int minX, int maxX){
	return max(minX, min(maxX, x));
}

extern "C" __global__ void advect(Array2D<0> q, Array2D<1> qNew, Array2D<2> u, Array2D<3> v, const float dt, const float rdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = qNew.getCount(0);
	int width = qNew.getCount(1);
	
	if (i >= width || j >= height || i < 0 || j < 0)
		return;

	float pos_x = i - u[j][i] * dt * rdx;
	float pos_y = j - v[j][i] * dt * rdx;
	pos_x = clamp(pos_x, 0.0, (float)width-1);
	pos_y = clamp(pos_y, 0.0, (float)height-1);
	int x = (int) floor(pos_x);
	int y = (int) floor(pos_y);
	float t_x = pos_x - x;
	float t_y = pos_y - y;


	// bilinear interpolation
	float pixel00 = q[y][x];
	float pixel10 = q[y][clamp((x + 1), 0, width - 1)];
	float pixel01 = q[clamp((y + 1), 0, height - 1)][x];
	float pixel11 = q[clamp((y + 1), 0, height - 1)][clamp((x + 1), 0, width - 1)];

	qNew[j][i] = (1.f - t_y)*((1.f - t_x)*pixel00 + t_x*pixel10) + t_y*((1.f - t_x)*pixel01 + t_x*pixel11);
}

extern "C" __global__ void jacobi(Array2D<0> x, Array2D<1> xNew, Array2D<2> b, const float alpha, const float rbeta)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = xNew.getCount(0);
	int width = xNew.getCount(1);

	if (i >= width || j >= height || i < 0 || j  < 0)
		return;

	xNew[j][i] = rbeta * (alpha * b[j][i]
		+ x[j][clamp((i + 1), 0, width - 1)]
		+ x[j][clamp((i - 1), 0, width - 1)]
		+ x[clamp((j + 1), 0, height - 1)][i]
		+ x[clamp((j - 1), 0, height - 1)][i]);
}

extern "C" __global__ void divergence(Array2D<0> u, Array2D<1> v, Array2D<2> div, const float halfrdx)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = div.getCount(0);
	int width = div.getCount(1);

	if (i >= width || j >= height || i < 0 || j < 0)
		return;

	div[j][i] = halfrdx * (u[j][clamp(i + 1, 0, width - 1)]
		- u[j][clamp(i - 1, 0, width - 1)]
		+ v[clamp(j + 1, 0, height - 1)][i]
		- v[clamp(j - 1, 0, height - 1)][i]);
}

extern "C" __global__ void subtractGradient(Array2D<0> p, Array2D<1> u, Array2D<2> v, Array2D<3> uNew, Array2D<4> vNew, const float halfrdx)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = uNew.getCount(0);
	int width = uNew.getCount(1);

	if (i >= width || j >= height || i < 0 || j < 0)
		return;

	uNew[j][i] = u[j][i] - halfrdx * (p[j][clamp(i + 1, 0, width - 1)]
		- p[j][clamp(i - 1, 0, width - 1)]);
	vNew[j][i] = v[j][i] - halfrdx * (p[clamp(j + 1, 0, height - 1)][i]
		- p[clamp(j - 1, 0, height - 1)][i]);
}


extern "C" __global__ void boundary(Array2D<0> x,  float scale)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = x.getCount(0);
	int width = x.getCount(1);

	if (i >= width || j >= height)
		return;

	if (i == 0)
		x[j][i] = scale*x[j][i + 1];
	else if (i == width - 1)
		x[j][i] = scale*x[j][i - 1];
	else if (j == 0)
		x[j][i] = scale*x[j + 1][i];
	else if (j == height - 1)
		x[j][i] = scale*x[j - 1][i];
}

extern "C" __global__ void addInk(Array2D<0> u, Array2D<1> v, Array2D<2> ink, const int x, const int y, const float u_, const float v_, const float ink_)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = u.getCount(0);
	int width = u.getCount(1);

	if (i >= width || j >= height || i < 0 || j < 0)
		return;
	
	int dx = i - x;
	int dy = j - y;
	float s = 1.f / pow(2., static_cast<double>(dx*dx + dy*dy) / 200.);

	u[j][i] += u_ * s;
	v[j][i] += v_ * s;
	ink[j][i] += ink_ * s;
	ink[j][i] = clamp(ink[j][i], 0.0, 255.0);
}

extern "C" __global__ void convertToColor(uint8_t *color, Array2D<0> x)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = x.getCount(0);
	int width = x.getCount(1);
	size_t index = i + width * j;

	if (i >= width || j >= height || i < 0 || j < 0)
		return;

	uint8_t value = 255 - static_cast<uint8_t>(x[j][i]);
	color[4 * index] = value;
	color[4 * index + 1] = value;
	color[4 * index + 2] = value;
	color[4 * index + 3] = 0;
}

extern "C" __global__ void convertToColor2(uint8_t *color, Array2D<0> r, Array2D<1> g, Array2D<2> b)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int height = r.getCount(0);
	int width = r.getCount(1);
	size_t index = i + width * j;

	if (i >= width || j >= height || i < 0 || j < 0)
		return;

	color[4 * index] = static_cast<uint8_t>(r[j][i]);
	color[4 * index + 1] = static_cast<uint8_t>(g[j][i]);
	color[4 * index + 2] = static_cast<uint8_t>(b[j][i]);
	color[4 * index + 3] = 0;
}