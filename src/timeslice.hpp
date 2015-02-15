#pragma once
#ifndef timeslice_hpp
#define timeslice_hpp
#include <chrono>
#include <iostream>
#include <atomic>
#include <string>
using namespace std;

class TimeSlice
{
public:
	TimeSlice(std::string MyName, bool Silent=false) : name(MyName), start(timestamp()), silent(Silent)
	{
		if (!silent)// I don' know if a compiler is smart enough to optomize this out, I do know its smart enough to do it when only fency is present
		{
			fency();//don't optomize me out!
		}
	}
	~TimeSlice()
	{
		if (!silent)
		{
			fency();
			auto elapsed = timestamp() - start;
			cout << name << " " << (int)elapsed << endl;
		}
	}
	static inline long long timestamp()
	{
		return chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
	}
private:
	const std::string name;
	const long long start;
	const bool silent;
	static inline void fency()
	{
		std::atomic_signal_fence(std::memory_order_seq_cst);
	}
};

#define TIMEME TimeSlice t(__FUNCTION__); 

#endif
