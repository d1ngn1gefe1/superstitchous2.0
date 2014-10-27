#include <algorithm>
#include "stitch.hpp"

//void getTile(const ImgFetcher &fetch, const Rect &roi, const Size &imSize, const Vector &xTrans, const Vector &yTrans, Mat &out) {
//	out.create(roi.size(), CV_32FC1);
//	assert(fetch.w * fetch.h == xTrans.size() + 1 && xTrans.size() == yTrans.size());
//	float x, y;
//	for (unsigned gridX = 0; gridX < fetch.w; gridX++) {
//		for (unsigned gridY = 0; gridY < fetch.h; gridY++) {
//			if (gridX == 0 && gridY == 0) {
//				continue;
//			}
//			unsigned ind = ptToI(gridX, gridY, fetch.w);
//			float x = xTrans[ind]
//			float y = yTrans[ind];
//
//			if (y > roi.y + roi.height) {
//				break;
//			}
//
//			if (!roi.contains({xTrans[i], yTrans[i]})
//					&& !roi.contains({xTrans[i] + imSize.width, yTrans[i]})
//					&& !roi.contains({xTrans[i], yTrans[i] + imSize.height})
//					&& !roi.contains({xTrans[i] + imSize.width, yTrans[i] + imSize.height})) {
//				break;
//			}
//
//			Mat im;
//			fetch.getMat({gridX, gridY}, im);
//		}
//		if (x > roi.x + roi.width) {
//			break;
//		}
//	}
//}

//// get absolute position in pixels of top-left pixel of image at coordinate (j, i)
//static void getPxPos(const LSQRVectors &trans, const GridPt &imPt, unsigned wInIms, Point2f &out) {
//	if (imPt[0] == 0 && imPt[1] == 0) {
//		out = Point2f(0, 0);
//	} else {
//		out.x = trans.xs[imPt[1] * wInIms + imPt[0] - 1];
//		out.y = trans.ys[imPt[1] * wInIms + imPt[0] - 1];
//	}
//}

// make pyramid shaped weight matrix in out
static void getWeightMat(const Size &imSz, Mat &out) {
	out.create(imSz, CV_32FC1);
	float slope = (float)imSz.height / imSz.width;
	for (float x = (float)(-(out.cols - 1) / 2.); x <= (out.cols - 1) / 2.; x++) {
		for (float y = (float)(-(out.rows - 1) / 2.); y <= (out.rows - 1) / 2.; y++) {
			int i = (int)(y + (out.rows - 1) / 2.);
			int j = (int)(x + (out.cols - 1) / 2.);
			if ((y > slope * x && y > -slope * x) || (y < slope * x && y < -slope * x)) {
				out.at<float>(i, j) = (float)(1 - abs(y) * 2. * (1 - 1e-10) / (out.rows - 1));
			} else {
				out.at<float>(i, j) = (float)(1 - abs(x) * 2. * (1 - 1e-10) / (out.cols - 1));
			}
		}
	}
}

TileHolder::TileHolder(const vector<string> &files, const vector<float> &xs, const vector<float> &ys, const Size &imSz, const Size &tileSz) :
		tileSz(tileSz) {
	assert(files.size() == xs.size() && xs.size() == ys.size());
//	for (int i = 0; i < files.size(); i++) {
//		printf("%s %f %f\n", files[i].c_str(), xs[i], ys[i]);
//	}

	auto xMinMax = minmax_element(xs.begin(), xs.end());
	float xMin = *(xMinMax.first), xMax = *(xMinMax.second);
	auto yMinMax = minmax_element(ys.begin(), ys.end());
	float yMin = *(yMinMax.first), yMax = *(yMinMax.second);

	tl = Point2f(xMin, yMin);
//	printf("abs tl: %f %f\n", tl.x, tl.y);

	szInTiles.width = (int)ceil((xMax + imSz.width - tl.x) / tileSz.width);
	szInTiles.height = (int)ceil((yMax + imSz.height - tl.y) / tileSz.height);
//	printf("size in tiles: %u %u\n", szInTiles.width, szInTiles.height);

	transBins.resize(szInTiles.area());

	for (int i = 0; i < files.size(); i++) {
		Point2f pos(xs[i], ys[i]);
		ImData dat = {files[i], pos};

		getBinFromPos(Point2f(xs[i], ys[i])).insert(dat);
		getBinFromPos(Point2f(xs[i] + imSz.width - 1, ys[i])).insert(dat);
		getBinFromPos(Point2f(xs[i], ys[i] + imSz.height - 1)).insert(dat);
		getBinFromPos(Point2f(xs[i] + imSz.width - 1, ys[i] + imSz.height - 1)).insert(dat);
	}

	getWeightMat(imSz, weightMat);
}

void TileHolder::makeTile(const GridPt &tilePt, ImgFetcher &fetch, Mat &out, Mat &bg) {
	Point2f tileTl(tl.x + tilePt[0] * tileSz.width, tl.y + tilePt[1] * tileSz.height);

	const set<ImData> &bin = getBin(tilePt);
	out = Mat::zeros(tileSz, CV_32FC1);
	Mat weightIm = Mat::zeros(tileSz, CV_32FC1) + FLT_MIN;//+ 1e-90;	// avoid divide by zero

	Mat curIm;
	for (const ImData &dat : bin) {
		fetch.getMat(dat.file, curIm);
		if (curIm.type() == CV_8U) {
			curIm.convertTo(curIm, CV_32F, 1. / 255);
		} else if (curIm.type() != CV_32F) {
			fprintf(stderr, "err: bad image type\n");
			exit(1);
		}

		if (bg.data)
		{
			assert(bg.size() == curIm.size());
			curIm -= bg;
		}

		blit(curIm.mul(weightMat), out, dat.pos - tileTl);
		blit(weightMat, weightIm, dat.pos - tileTl);
	}
	out /= weightIm;
}
