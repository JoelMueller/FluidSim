Disclaimer: The following Readme is valid for running the application on the hhlr.

1 Introduction
This fluidsimulation serves as a bechmark for the MATOG Auto-Tuning framework.
It is based on the article Fast Fluid Dynamics Simulation on the GPU from NVIDIA GPU Gems. http://http.developer.nvidia.com/GPUGems/gpugems_ch38.html

1.1 Requirements
Without GUI
MATOG 4.0
CUDA 6.0 or higher
GCC 4.8 or higher
CMake 2.8 or higher
OpenCV 3.0 or higher
Intel TBB 4.4

With GUI (additionally)
GLEW
GLFW

2 Compiling
mkdir build
cd build
cmake ..
make
make copy

To compile the application with GUI activate the GUI option in cmake.
(This was only tested using Windows10 and VisualStudio 13)

3 Running
Usage: ./fluidsim <option(s)>
Options:
    -h,--help                       Show this help message
    -g,--gif                        Save simulation as gif
    -s,--size       WIDTH HEIGHT    Specify simulation size
    -p,--pre        PATH            Specify predefined user interaction

If the option -g is used the results are saved to ink.gif and p.gif in the working folder.
job.sh is preconfigured to run a test sample.

4 Predefined UserInput
To simulate user input the application reads files with following pattern:
Each line has following structure:

frame_start frame_end   x_start     x_end       y_start     y_end       intensity
integer     integer     float[0,1]  float[0,1]  float[0,1]  float[0,1]  float

Frames in different lines are assumed to be not overlapping.
The application assumes the files to be correct and performs no sanity checks.
In the folder "sim_interaction" you can find two predefined use cases.
interaction01: runs for 1000 iterations with some predefined events
interaction02: runs for 10 iterations and can be used in the profiling step of matog.
