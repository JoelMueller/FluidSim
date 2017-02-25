/*
 * Based on https://gist.github.com/tautologico/2879581
 * Andrei de A. Formiga, 2012-06-04
 */

#include <iostream>

#include "fluidsimulation.h"

static void show_usage(std::string name)
{
	std::cout << "Usage: " << name << " <option(s)>\n"
		<< "Options:\n"
		<< "\t-h,--help\t\t\tShow this help message\n"
		<< "\t-g,--gif\t\t\tSave simulation as gif\n"
		<< "\t-s,--size\tWIDTH HEIGHT\tSpecify simulation size\n"
		<< "\t-p,--pre\tPATH\t\tSpecify predefined user interaction, disables GUI\n"
        << "\t-t,--threads\tTHREADS_X THREADS_Y\tSpecfiy the number of threads per block\n"
		<< std::endl;
}

int main(int argc, char **argv)
{
	int width = 512;
	int height = 512;
    int threads_x = 16;
    int threads_y = 16;
	bool saveImages = false;
	bool predefined = false;
	const char* userdata_path = "";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help")) {
			show_usage(argv[0]);
			return 0;
		}
		else if ((arg == "-g") || (arg == "--gif")) {
			saveImages = true;
		}
        else if ((arg == "-t") || (arg == "--threads")) {
			if (i + 2 < argc) {
				sscanf(argv[++i], "%i", &threads_x);
				sscanf(argv[++i], "%i", &threads_y);
			}
			else {
				std::cout << "--threads option requires two arguments." << std::endl;
				return 1;
			}
		}
		else if ((arg == "-s") || (arg == "--size")) {
			if (i + 2 < argc) {
				sscanf(argv[++i], "%i", &width);
				sscanf(argv[++i], "%i", &height);
			}
			else {
				std::cout << "--size option requires two arguments." << std::endl;
				return 1;
			}
		}
		else if ((arg == "-p") || (arg == "--predefined")) {
			if (i + 1 < argc) {
				userdata_path = argv[++i];
				predefined = true;
			}
			else {
				std::cout << "--predefined option requires one argument." << std::endl;
				return 1;
			}
		}
		else {
			show_usage(argv[0]);
			return 1;
		}
	}

	FluidSimulation fluidSim(width, height, threads_x, threads_y, saveImages, userdata_path);

    return 0;
}
