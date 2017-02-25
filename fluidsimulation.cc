#define _USE_MATH_DEFINES

#ifdef _WIN32
#include <Windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>

#include "fluidsimulation.h"
#include "gif.h"
#include "timer.h"
#include "common.h"
#include <iostream>

#ifdef WITH_GUI
#include <cudaGL.h>
#endif


const char *module_file = (char*) "fluidSimKernel.cu";
const char *advection_kernel_name = (char*) "advect";
const char *jacobi_kernel_name = (char*) "jacobi";
const char *divergence_kernel_name = (char*) "divergence";
const char *subtractGradient_kernel_name = (char*) "subtractGradient";
const char *boundary_kernel_name = (char*) "boundary";
const char *addInk_kernel_name = (char*) "addInk";
const char *convertToColor_kernel_name = (char*) "convertToColor";
const char *convertToColor2_kernel_name = (char*) "convertToColor2";

const char *fileP = "p.gif";
const char *fileInk = "ink.gif";

using namespace FluidSim;

FluidSimulation::FluidSimulation(int width /*= 512*/, int height /*= 512*/, int threads_x /*= 16*/, int threads_y /*= 16*/, bool saveImages_ /*= false*/, const char * inputFile /*= ""*/)
	: gifWriterP(new GifWriter)
	, gifWriterInk(new GifWriter)
	, lastPosX(-1)
	, lastPosY(-1)
	, currentColor(RED)
	, saveImages(saveImages_)
	, eventIndex(0)
{
	printf("- Initializing...\n");

	info.width = width;
	info.height = height;
    info.threads_x = threads_x;
    info.threads_y = threads_y;
    
	// check for empty string
	predefined = (inputFile && inputFile[0] != '\0');

	if (predefined);
		loadEventsFromFile(inputFile);

	initCUDA();
	initGL();

	setupDeviceMemory();
	setupHostMemory();

	copyAllHtoD();

	cuCtxSynchronize();

	int i = 0;
#ifndef WITH_GUI
	gifWriterP.reset(new GifWriter);
	gifWriterInk.reset(new GifWriter);
    startWritingToImage();
    
	for(; true; i++){
		if (predefined && eventIndex >= events.size()){
            break;
        }
        update(i);        
    }		

	stopWritingToImage();
#else
	// rendering loop
	do
	{
		glClear(GL_COLOR_BUFFER_BIT);

		update(i++);

		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&	glfwWindowShouldClose(window) == 0);
#endif
}

FluidSimulation::~FluidSimulation()
{
	printf("- Finalizing...\n");

	releaseDeviceMemory();
	releaseHostMemory();
	
	cuCtxDetach(info.context);

#ifdef WITH_GUI
	// Close OpenGL window
	glfwTerminate();
#endif
}

void FluidSimulation::loadEventsFromFile(const char* path)
{
	std::ifstream infile(path);
	for (std::string line; std::getline(infile, line);)
	{
		std::istringstream in(line);

		int frame_start, frame_end;
		float x_start, x_end, y_start, y_end, amount;
		in >> frame_start >> frame_end >> x_start >> x_end >> y_start >> y_end >> amount;

		UserEvent event;
		event.frame_start = frame_start;
		event.frame_end = frame_end;
		event.x_start = clamp(x_start, 0.0, 1.0)*info.width;
		event.x_end = clamp(x_end, 0.0, 1.0)*info.width;
		event.y_start = clamp(y_start, 0.0, 1.0)*info.height;
		event.y_end = clamp(y_end, 0.0, 1.0)*info.height;
		event.amount = amount;
		events.push_back(event);
	}
}

void FluidSimulation::initGL()
{
#ifdef WITH_GUI
	// Initialise GLFW
	if (!glfwInit())
		fprintf(stderr, "Failed to initialize GLFW\n");

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(info.width, info.height, "Fluid Simulation", NULL, NULL);
	if (!window)
	{
		fprintf(stderr, "Failed to open GLFW window\n");
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		glfwTerminate();
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// create texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &outputTexture);

	glBindTexture(GL_TEXTURE_2D, outputTexture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, info.width, info.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	// register outputTexture to cuda graphics resource
	checkCudaErrors(cuGraphicsGLRegisterImage(&outputTextureResource, outputTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
#endif
}

void FluidSimulation::initCUDA()
{
	int deviceCount = 0;
	CUresult err = cuInit(0);
	int major = 0, minor = 0;

	if (err == CUDA_SUCCESS)
		CHECK(cuDeviceGetCount(&deviceCount));

	if (deviceCount == 0)
	{
		fprintf(stderr, "Error: no devices supporting CUDA\n");
		exit(-1);
	}

	auto& device = info.device;
	auto& totalGlobalMem = info.totalGlobalMem;
	auto& context = info.context;
	auto& module = info.module;

	// get first CUDA device
	CHECK(cuDeviceGet(&device, 0));
	char name[100];
	cuDeviceGetName(name, 100, device);
	printf("> Using device 0: %s\n", name);

	// get compute capabilities and the devicename
	CHECK(cuDeviceComputeCapability(&major, &minor, device));
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	CHECK(cuDeviceTotalMem(&totalGlobalMem, device));
	printf("  Total amount of global memory:   %llu bytes\n",
		(unsigned long long)totalGlobalMem);
	printf("  64-bit Memory Address:           %s\n",
		(totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ?
		"YES" : "NO");

	CHECK(cuCtxCreate(&context, 0, device));
	CHECK(cuModuleLoad(&module, module_file));
	CHECK(cuModuleGetFunction(&info.advection_function, module, advection_kernel_name));
	CHECK(cuModuleGetFunction(&info.jacobi_function, module, jacobi_kernel_name));
	CHECK(cuModuleGetFunction(&info.divergence_function, module, divergence_kernel_name));
	CHECK(cuModuleGetFunction(&info.subtractGradient_function, module, subtractGradient_kernel_name));
	CHECK(cuModuleGetFunction(&info.boundary_function, module, boundary_kernel_name));
	CHECK(cuModuleGetFunction(&info.addInk_function, module, addInk_kernel_name));
	CHECK(cuModuleGetFunction(&info.convertToColor_function, module, convertToColor_kernel_name));
	CHECK(cuModuleGetFunction(&info.convertToColor2_function, module, convertToColor2_kernel_name));
}

void FluidSimulation::setupDeviceMemory()
{
	auto height = info.height;
	auto width = info.width;
	d_u = new Array2D::Device(height, width, _fl);
	d_v = new Array2D::Device(height, width, _fl);
	d_temp1 = new Array2D::Device(height, width, _fl);
	d_temp2 = new Array2D::Device(height, width, _fl);
	d_p = new Array2D::Device(height, width, _fl);
	d_ink_r = new Array2D::Device(height, width, _fl);
	d_ink_g = new Array2D::Device(height, width, _fl);
	d_ink_b = new Array2D::Device(height, width, _fl);
	CHECK(cuMemAlloc(&d_image, 4*sizeof(uint8_t) * info.height*info.width));
}

void FluidSimulation::setupHostMemory()
{
	auto height = info.height;
	auto width = info.width;
	u = new Array2D::Host<>(height, width, _fl);
	v = new Array2D::Host<>(height, width, _fl);
	temp1 = new Array2D::Host<>(height, width, _fl);
	temp2 = new Array2D::Host<>(height, width, _fl);
	p = new Array2D::Host<>(height, width, _fl);
	ink_r = new Array2D::Host<>(height, width, _fl);
	ink_g = new Array2D::Host<>(height, width, _fl);
	ink_b = new Array2D::Host<>(height, width, _fl);
	image.resize(4 * info.height * info.width, 0);
}

void FluidSimulation:: initHostMemory()
{
	auto height = info.height;
	auto width = info.width;
	// initialize host arrays
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x){
			(*u)[y][x] = 0.f;
			(*v)[y][x] = 0.f;
			(*temp1)[y][x] = 0.f;
			(*temp2)[y][x] = 0.f;
			(*p)[y][x] = 0.f;
			(*ink_r)[y][x] = 0.f;
			(*ink_g)[y][x] = 0.f;
			(*ink_b)[y][x] = 0.f;
			image[4 * (x + y*width)] = 0;
			image[4 * (x + y*width) + 1] = 0;
			image[4 * (x + y*width) + 2] = 0;
			image[4 * (x + y*width) + 3] = 0;
		}
	}
}

void FluidSimulation::releaseDeviceMemory()
{
	CHECK(cuMemfree(d_image));
}

void FluidSimulation::releaseHostMemory()
{
	delete u;
	delete v;
	delete temp1;
	delete temp2;
	delete p;
	delete ink_r;
	delete ink_g;
	delete ink_b;
}

void FluidSimulation::copyAllHtoD()
{
	CHECK(cuMemcpyHtoD(d_u, u, 0));
	CHECK(cuMemcpyHtoD(d_v, v, 0));
	CHECK(cuMemcpyHtoD(d_temp1, temp1, 0));
	CHECK(cuMemcpyHtoD(d_temp2, temp2, 0));
	CHECK(cuMemcpyHtoD(d_p, p, 0));
	CHECK(cuMemcpyHtoD(d_ink_r, ink_r, 0));
	CHECK(cuMemcpyHtoD(d_ink_g, ink_g, 0));
	CHECK(cuMemcpyHtoD(d_ink_b, ink_b, 0));
}

void FluidSimulation::copyAllDtoH()
{
	CHECK(cuMemcpyHtoD(d_u, u, 0));
	CHECK(cuMemcpyHtoD(d_v, v, 0));
	CHECK(cuMemcpyHtoD(d_temp1, temp1, 0));
	CHECK(cuMemcpyHtoD(d_temp2, temp2, 0));
	CHECK(cuMemcpyHtoD(d_p, p, 0));
	CHECK(cuMemcpyHtoD(d_ink_r, ink_r, 0));
	CHECK(cuMemcpyHtoD(d_ink_g, ink_g, 0));
	CHECK(cuMemcpyHtoD(d_ink_b, ink_b, 0));
}

void FluidSimulation::renderImage()
{
#ifdef WITH_GUI
	auto width = info.width;
	auto height = info.height;

	CHECK(cuGraphicsMapResources(1, &outputTextureResource, 0));
	CUarray textureArray;
	CHECK(cuGraphicsSubResourceGetMappedArray(&textureArray, outputTextureResource, 0, 0));

	// write color value of ink to image
	//convertToColor(info, d_image, d_ink);
	convertToColor2(info, d_image, d_ink_r, d_ink_g, d_ink_b);

	CUDA_MEMCPY2D memcpy;
	memcpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	memcpy.srcPitch = 4 * width;
	memcpy.srcDevice = d_image;
	memcpy.srcXInBytes = 0;
	memcpy.srcY = 0;
	memcpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	memcpy.dstArray = textureArray;
	memcpy.dstXInBytes = 0;
	memcpy.dstY = 0;
	memcpy.WidthInBytes = 4 * width;
	memcpy.Height = height;

	CHECK(cuMemcpy2D(&memcpy));

	CHECK(cuGraphicsUnmapResources(1, &outputTextureResource, 0));

	glViewport(0, 0, width, height);

	glBindTexture(GL_TEXTURE_2D, outputTexture);
	{
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
			glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
			glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
			glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		}
		glEnd();
	}
	glBindTexture(GL_TEXTURE_2D, 0);
#endif
}

void FluidSimulation::saveImagesAsGif()
{
	if (!saveImages)
		return;

	cuCtxSynchronize();
	CHECK(cuMemcpyDtoH(p, d_p, 0));
	CHECK(cuMemcpyDtoH(ink_r, d_ink_r, 0));
	CHECK(cuMemcpyDtoH(ink_g, d_ink_g, 0));
	CHECK(cuMemcpyDtoH(ink_b, d_ink_b, 0));
	cuCtxSynchronize();

	writePressureToImage();
	if (!GifWriteFrame(gifWriterP.get(), image.data(), info.width, info.height, 4))
		fprintf(stderr, "Error writing %s!\n", fileP);

	writeInkToImage();
	if (!GifWriteFrame(gifWriterInk.get(), image.data(), info.width, info.width, 4))
		fprintf(stderr, "Error writing %s!\n", fileInk);
}

void FluidSimulation::checkForUserInput()
{
#ifdef WITH_GUI
	bool leftClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	bool rightClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
		currentColor = RED;
	else if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
		currentColor = GREEN;
	else if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
		currentColor = BLUE;

	// reset
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		setupHostMemory();
		copyAllHtoD();
	}

	if (leftClick || rightClick)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		int currentPosX = static_cast<int>(xpos);
		int currentPosY = static_cast<int>(ypos);

		if (lastPosX >= 0 && lastPosY >= 0)
		{
			// determine velocity of cursor movement
			float factor = 50.f;
			float velocityX = static_cast<float>(currentPosX - lastPosX) * factor;
			float velocityY = static_cast<float>(currentPosY - lastPosY) * factor;
			float inkToAdd = leftClick ? 100.f : 0.f;

			// determine color choice
			Array2D::Device* d_ink;
			switch (currentColor)
			{
			case RED:
				d_ink = d_ink_r;
				break;
			case GREEN:
				d_ink = d_ink_g;
				break;
			case BLUE:
				d_ink = d_ink_b;
				break;
			}

			addInk(info, d_u, d_v, d_ink, currentPosX, HEIGHT - currentPosY, velocityX, -velocityY, inkToAdd);
		}

		lastPosX = currentPosX;
		lastPosY = currentPosY;
	}
	else
	{
		lastPosX = -1;
		lastPosY = -1;
	}
#endif
}

void FluidSimulation::predefinedInput(int iteration)
{
	auto & e = events[eventIndex];
	if (iteration >= e.frame_start)
	{
		if (iteration < e.frame_end){
			InkData data = getInkData(e, iteration);
            addInk(info, d_u, d_v, d_ink_r, data.x, data.y, data.u, data.v, data.amount);
        }
		else
			eventIndex++;
	}
}

void FluidSimulation::predefinedScenario(int iteration, Scenario s)
{
	auto width = info.width;
	auto height = info.height;

	if (iteration % 10 != 0)
		return;

	int x = width / 2;
	int y = height / 2;

	switch (s)
	{
	case CONSTANT:
		if (iteration % 10 != 0)
			return;
		addInk(info, d_u, d_v, d_ink_r, x, y, 100.f, 0.f, 50.f);
		addInk(info, d_u, d_v, d_ink_g, x, y, 0.f, 0.f, 30.f);
		addInk(info, d_u, d_v, d_ink_b, x, y, 0.f, 0.f, 10.f);
		break;
	case RANDOM:
		addInk(info, d_u, d_v, d_ink_r, rand() % width, rand() % height, (rand() % 400) - 200, (rand() % 400) - 200, 100.f);
		addInk(info, d_u, d_v, d_ink_g, rand() % width, rand() % height, (rand() % 400) - 200, (rand() % 400) - 200, 100.f);
		addInk(info, d_u, d_v, d_ink_b, rand() % width, rand() % height, (rand() % 400) - 200, (rand() % 400) - 200, 100.f);
		break;
	case ALTERNATING:
		addInk(info, d_u, d_v, d_ink_r, x, y, 100.f, 100.f*sin(static_cast<float>(iteration) / 300.f * M_PI), 50.f);
		addInk(info, d_u, d_v, d_ink_g, x, y, 0.f, 0.f, 30.f);
		addInk(info, d_u, d_v, d_ink_b, x, y, 0.f, 0.f, 10.f);
		break;
	}
}

FluidSimulation::InkData FluidSimulation::getInkData(UserEvent & e, int frame)
{
	InkData data;
	float t = (frame - e.frame_start) / (e.frame_end - e.frame_start);
	data.x = e.x_start * (1 - t) + (e.x_end) * t;
	data.y = e.y_start * (1 - t) + (e.y_end) * t;
	data.u = 10*(e.x_end - e.x_start) / (e.frame_end - e.frame_start);
	data.v = 10*(e.y_end - e.y_start) / (e.frame_end - e.frame_start);
	data.amount = e.amount;
	return data;
}

void FluidSimulation::update(int i)
{
	// constants
	int poissonSteps = 35;
	float dt = 0.001f;
	float dx = 0.1f;
	float viscosity = 0.001f;
	float ink_longevity = 0.001f;

	float rdx = 1.f / dx;
	float halfrdx = 0.5f*rdx;
	float alpha_d = dx*dx / (viscosity*dt);
	float rbeta_d = 1.f / (4.f + alpha_d);
	float alpha_p = -dx*dx;
	float rbeta_p = 1.f / 4.f;

	// no-slip velocity boundary condition
	boundary(info, d_u, -1);
	boundary(info, d_v, -1);
	boundary(info, d_ink_r, 0);
	boundary(info, d_ink_g, 0);
	boundary(info, d_ink_b, 0);

	// advection
	advect(info, d_u, d_temp1, d_u, d_v, dt, rdx);
	advect(info, d_v, d_temp2, d_u, d_v, dt, rdx);
	std::swap(d_u, d_temp1);
	std::swap(d_v, d_temp2);
	advect(info, d_p, d_temp1, d_u, d_v, dt, rdx);
	std::swap(d_p, d_temp1);
	advect(info, d_ink_r, d_temp1, d_u, d_v, dt, rdx);
	std::swap(d_ink_r, d_temp1);
	advect(info, d_ink_g, d_temp1, d_u, d_v, dt, rdx);
	std::swap(d_ink_g, d_temp1);
	advect(info, d_ink_b, d_temp1, d_u, d_v, dt, rdx);
	std::swap(d_ink_b, d_temp1);

	// apply force and add ink
#ifdef WITH_GUI
	checkForUserInput();
#else
	if (predefined && eventIndex < events.size())
		predefinedInput(i);
	else
		predefinedScenario(i, ALTERNATING);
#endif

	// diffusion
	for (int i = 0; i < poissonSteps; ++i)
	{
		jacobi(info, d_u, d_temp1, d_u, alpha_d, rbeta_d);
		jacobi(info, d_v, d_temp2, d_v, alpha_d, rbeta_d);
		std::swap(d_u, d_temp1);
		std::swap(d_v, d_temp2);
	}

	// projection into divergence-free field
	divergence(info, d_u, d_v, d_temp1, halfrdx);
	for (int i = 0; i < poissonSteps; ++i)
	{
		boundary(info, d_p, 1);
		jacobi(info, d_p, d_temp2, d_temp1, alpha_p, rbeta_p);
		std::swap(d_p, d_temp2);
	}

	boundary(info, d_u, -1);
	boundary(info, d_v, -1);

	subtractGradient(info, d_p, d_u, d_v, d_temp1, d_temp2, halfrdx);
	std::swap(d_u, d_temp1);
	std::swap(d_v, d_temp2);

#ifdef WITH_GUI
	renderImage();
#else
	if (i % 10 == 0)
		saveImagesAsGif();
#endif
}

void FluidSimulation::startWritingToImage()
{
	if (!saveImages)
		return;

	GifBegin(gifWriterP.get(), fileP, info.width, info.height, 4);
	GifBegin(gifWriterInk.get(), fileInk, info.width, info.height, 4);
}

void FluidSimulation::stopWritingToImage()
{
	if (!saveImages)
		return;

	GifEnd(gifWriterP.get());
	GifEnd(gifWriterInk.get());
}

void FluidSimulation::writePressureToImage()
{
	for (int y = 0; y < info.height; ++y) for (int x = 0; x < info.width; ++x)
	{
		float p_ = (*p)[y][x];
		int j = x + y*info.width;

		if (p_ < 0) // negative values are blue
		{
			auto value = static_cast<uint8_t>(std::min(-p_ * 50.f, 255.f));
			image[4 * j] = 0;
			image[4 * j + 1] = 0;
			image[4 * j + 2] = value;
			image[4 * j + 3] = value;
		}
		else // positive values are red
		{
			auto value = static_cast<uint8_t>(std::min(p_ * 50.f, 255.f));
			image[4 * j] = value;
			image[4 * j + 1] = 0;
			image[4 * j + 2] = 0;
			image[4 * j + 3] = value;
		}
	}
}

void FluidSimulation::writeInkToImage()
{
	for (int y = 0; y < info.height; ++y) for (int x = 0; x < info.width; ++x)
	{
		int j = x + y*info.width;
		
		image[4 * j] = static_cast<uint8_t>((*ink_r)[y][x]);
		image[4 * j + 1] = static_cast<uint8_t>((*ink_g)[y][x]);
		image[4 * j + 2] = static_cast<uint8_t>((*ink_b)[y][x]);
		image[4 * j + 3] = 0;
	}
}