#pragma once

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <exception>
#include <vector>
#include <string>
#include <utility>
#include "opencv2/opencv.hpp"
#include "fetcher.hpp"
#include "fftw3.h"
#include "common.hpp"
#include <limits>       // std::numeric_limits

using namespace std;
using namespace cv;

struct MaxDists {
	int x;	// max peak err for horizontal offset
	int y;	// vertical offset
	int xy;	// diagonal offset
	friend ostream& operator<<(ostream& os, const MaxDists& a)
	{
		return os << "(" << a.x << "," << a.y << "," << a.xy << ")";
	}

};

static inline unsigned getFFTLen(const Size& imSz) {
	return imSz.height * (imSz.width + 2);
}

typedef float (*ConfGetter)(const Mat &peakIm, const Point &maxLoc, double bestSqDist);

extern int peakRadius;

Point phaseCorr(const float *fix, const float *im, const Size &imSz, const fftwf_plan &plan, float *buf, const Point2f &hint,
		double &bestSqDist, const Mat &mask, Point &maxLoc, float &conf, const ConfGetter getConf, const char *saveIm = NULL);
float *getFFT(const void *matData, const Size &imSz, const fftwf_plan &plan);

// store translations into transMap
void storeTrans(ImgFetcher &fetcher, const Point2f &absHint, PairToTransData &transMap, const MaxDists &dists);

void writeTranMap(const string &file, const PairToTransData &transMap);
void readTranMap(const string &file, PairToTransData &transMap);

// remove outliers
void rmOutliers(PairToTransData &transMap);
