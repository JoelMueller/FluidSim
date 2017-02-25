#pragma once

#include <Matog.h>
#include <cstdint>
#include <vector>
#include <memory>

#ifdef WITH_GUI
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include "matog_gen/Array2D.h"

#include "fluidSimKernel.h"

struct GifWriter;

class FluidSimulation
{
	enum Scenario
	{
		CONSTANT, 	// constant stream in the center of the image, shooting ink to the right
		RANDOM,		// ink gets added randomly somewhere inside the image in a random direction
		ALTERNATING	// ink gets shot from the center of the image to the right, alternating up and down
	};

	enum ColorMode
	{
		RED,
		GREEN,
		BLUE
	};

	struct InkData
	{
		int x;
		int y;
		float u;
		float v;
		int amount;
	};

	struct UserEvent
	{
		int frame_start;
		int frame_end;
		float x_start;
		float x_end;
		float y_start;
		float y_end;
		float amount;
	};

public:

	FluidSimulation(int width = 512, int height = 512, int threads_x = 16, int threads_y = 16, bool saveImages_ = false, const char * inputFile = "");
	~FluidSimulation();

private:

	// main iteration function
	void update(int i);

	void initGL();
	void initCUDA();

	void setupDeviceMemory();
	void setupHostMemory();
	void initHostMemory();
	void releaseDeviceMemory();
	void releaseHostMemory();

	void copyAllHtoD();
	void copyAllDtoH();

	// output functions
	void renderImage();
	void saveImagesAsGif();
	void startWritingToImage();
	void stopWritingToImage();
	void writePressureToImage();
	void writeInkToImage();

	// input functions
	void checkForUserInput();
	void predefinedInput(int iteration);
	void predefinedScenario(int iteration, Scenario s);
	
	// load predefined input sequence from file
	void loadEventsFromFile(const char* path);
	InkData getInkData(UserEvent & e, int frame);

	// struct containing all necessary information about the device and the data
	FluidSim::cudaInfo info;

	// host pointers to data
	Array2D::Host<>* u, * v, * temp1, * temp2, * p, * ink_r, * ink_g, * ink_b;
	// device pointers to data
	Array2D::Device* d_u, * d_v, * d_temp1, * d_temp2, * d_p, * d_ink_r, * d_ink_g, * d_ink_b;

	// image data
	std::vector<uint8_t> image;
	CUdeviceptr d_image;

#ifdef WITH_GUI
	GLFWwindow* window;
	GLuint outputTexture;
	CUgraphicsResource outputTextureResource;
#endif

	// events from predefined input sequence
	std::vector<UserEvent> events;
	int eventIndex;

	int lastPosX, lastPosY;

	ColorMode currentColor;

	bool saveImages;
	bool predefined;

	std::unique_ptr<GifWriter> gifWriterP, gifWriterInk;
};