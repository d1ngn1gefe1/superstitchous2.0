#pragma once

#include <array>
#include <map>
#include "opencv2/opencv.hpp"
#include <cassert>
#include <vector>

using namespace std;
using namespace cv;

//#define IMRW_BUG

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef CMP
#define CMP(a, b) ((a) < (b) ? -1 : ((a) > (b) ? 1 : 0))
#endif

// use array so that it can be compared (and put into maps)
typedef array<int, 2> Pt;
typedef Pt GridPt;
typedef Pt PxPoint;
typedef Pt CorePt;

typedef array<GridPt, 2> PtPair; // pair of coordinates of 2 tile images

// a translation between adjacent tiles.
//
// `dist` is pixel distance (squared) between `trans` and hint
// `conf` is confidence of translation (eg. intensity of phase correlation peak)
struct TransData {
	Point trans;
	double dist;
	float conf;

	TransData();
	TransData(int x, int y, double dist, float conf);
};

typedef map<PtPair, TransData> PairToTransData; // coordinates of 2 tile images and the corresponding translation

template <class T, class U>
static inline double getSqDist(const Point_<T> &a, const Point_<U> &b) {
	double dx = a.x - b.x, dy = a.y - b.y;
	return dx * dx + dy * dy;
}

static inline Pt makePt(int x, int y) {
	Pt pt = {{x, y}};
	return pt;
}

static inline PtPair makePair(const GridPt& p1,const GridPt& p2) {
	PtPair p = {{p1, p2}};
	return p;
}

// return true if found, also stores the pair found into `pair` and value into `out`, and if isSwap provided,
// store if pair had to be swapped
template <class T>
static bool lookupPair(const map<PtPair, T> &pairToT, PtPair &pair, T& out, bool *isSwap=NULL) {
	auto it = pairToT.find(pair);
	if (it != pairToT.end()) {
		out = it->second;
		if (isSwap) *isSwap = false;
		return true;
	}
	PtPair swapped = {{pair[1], pair[0]}};
	it = pairToT.find(swapped);
	if (it != pairToT.end()) {
		pair = swapped;
		out = it->second;
		if (isSwap) *isSwap = true;
		return true;
	}
	return false;
}

// get current time
string getCurTime();

// imwrite
void imWriteWin(const string &name, const Mat &im);

// save a CV_32FC1 or CV_64FC1 image
void saveFloatIm(const string &name, const Mat &im);

// save and normalize with minPx and maxPx
void saveFloatIm(const string &name, const Mat &im, float minPx, float maxPx);

// save tiff in 32-bit float format
void saveFloatTiff(const string &name, const Mat &im);

void loadFloatTiff(const string &name, Mat &im, int w, int h);

void loadFloatTiffFast(const string &name, Mat &im, int w, int h);

// Blit a onto b at position tl (additively). tl is relative to top-left of b.
void blit(const Mat &a, Mat &b, Point tl);

// Crop a image 
void imCrop(const string &outPath, const string &inPath, const Rect& outRec, const Size& tileSz, const Size& gridDim, const string &fileName);

// create a downsampled image
void saveDownSampleIm(const string &path, Size tileSz, Size gridDim, int scaleLevel, Mat &outIm);

// background subtraction
void getBg(Mat &bgIm, vector<string> &imPaths, float bgSub, int w, int h);

// write to a file
FILE *fopenWrap(const char *path, const char *mode);