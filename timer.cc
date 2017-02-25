#include "timer.h"

void Timer::tic()
{
	lastTimePoint = high_res_clock::now();
}

long long Timer::toc()
{
	duration d = high_res_clock::now() - lastTimePoint;
	totalTime += d;
	return std::chrono::duration_cast<ms>(d).count();
}

void Timer::reset()
{
	totalTime = duration(0);
}

long long Timer::getTotalTime()
{
	return std::chrono::duration_cast<ms>(totalTime).count();
}
