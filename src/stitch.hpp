#pragma once
#include <cassert>
#include <vector>
#include <set>
#include "opencv2/opencv.hpp"
#include "lsqr_int.hpp"
#include "fetcher.hpp"
#include "optimize.hpp"

using namespace std;
using namespace cv;

struct ImData {
	string file;
	Point2f pos;
};

inline bool operator<(const ImData &lhs, const ImData &rhs) {
	return lhs.file < rhs.file;
}

struct TileHolder {
	vector<set<ImData>> transBins;	// holds a set of image data for each tile, initialized in constructor
	Size szInTiles; // grid size of output images
	Size tileSz;
	Point2f tl;

	Mat weightMat;	// 32F type

	TileHolder(const vector<string> &files, const vector<float> &xs, const vector<float> &ys, const Size &imSz, const Size &tileSz);

	inline set<ImData> &getBin(const GridPt &tilePt) {
		return transBins[tilePt[1] * szInTiles.width + tilePt[0]];
	}

	inline set<ImData> &getBinFromPos(const Point2f &pos) {
		assert(pos.x >= tl.x && pos.y >= tl.y);
		GridPt pt = {{(int)((pos.x - tl.x) / tileSz.width), (int)((pos.y - tl.y) / tileSz.height)}};
		return getBin(pt);
	}

	void makeTile(const GridPt &tilePt, ImgFetcher &fetch, Mat &out, Mat &bg = Mat());
};
