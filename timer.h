#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock high_res_clock;
typedef high_res_clock::time_point timePoint;
typedef high_res_clock::duration duration;
typedef std::chrono::milliseconds ms;

class Timer
{
public:
	void tic();
	long long toc();

	void reset();

	long long getTotalTime();

private:
	timePoint lastTimePoint;
	duration totalTime;
};