#pragma once
#include <array>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "common.hpp"
#include "lrucache.hpp"

using namespace std;
using namespace cv;

//enum class ImgPattern {
//	SNAKE_BY_ROW,
//	SNAKE_BY_COL
//};

// return the index of the snake-shaped image array
//unsigned getImgI(ImgPattern pat, GridPt pt, const Size &szInIms);

struct ImgFetcher {
	vector<string> imgPaths;
	Size szInIms;
	bool row_major;
	unsigned cap;
	Size imSz;
	bool useFastRead;
	bool fixNaNs;

	LRUCache<string, Mat> cache;

	ImgFetcher(const vector<string> &imgPaths, Size szInIms, size_t cacheSz,  bool fixNaNs);

	void getMat(const string &file, Mat &out);
	void getMat(const GridPt &pt, Mat &out);
};
