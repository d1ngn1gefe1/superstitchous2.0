#define _CRT_SECURE_NO_WARNINGS

#include "common.hpp"
#include "opencv2/opencv.hpp"
#include <set>
#include <array>
#include <string>
#include <cmath>
#include <fstream>
#include <direct.h>
#include <numeric>

using namespace std;
using namespace cv;

enum class CatType {
	Vertical,
	Horizontal
};

template <class T>
void printVec(const vector<T> &v) {
	cout << "{";
	for (size_t i = 0; i < v.size(); i++) {
		cout << v[i] << ", ";
	}
	cout << "}" << endl;
}

int sqDist(Point a, Point b) {
	int dx = a.x - b.x;
	int dy = a.y - b.y;
	return dx * dx + dy * dy;
}

Point descend(const Size &winSz, const Mat &im, const Point &initTl, bool &converged, bool checkMin = true) {
	Rect roi(initTl, winSz);
	Rect oldRoi(0, 0, 0, 0);
	do {
		oldRoi = roi;

		Moments mom(moments(im(roi), true));
		if (mom.m10 == 0 && mom.m01 == 0 && mom.m00 == 0) {
			converged = false;
			return roi.tl();
		}

		roi.x += (int)(mom.m10 / mom.m00 - winSz.width / 2.);
		roi.y += (int)(mom.m01 / mom.m00 - winSz.height / 2.);

		//		imshow("i", im(roi));
		//		waitForKeypress();
	} while (roi != oldRoi);

	Point oldTl = roi.tl();
	bool myConv;
	if (checkMin) {
		for (int dx = -1; dx < 2; dx++) {
			for (int dy = -1; dy < 2; dy++) {
				if (sqDist(descend(winSz, im, Point(roi.x + dx, roi.y + dy), myConv, false), oldTl) > 1) {
					converged = false;
					return roi.tl();
				}
			}
		}
	}

	converged = true;
	return roi.tl();
}

void pad(const Mat &im, Mat &out, int xPad, int yPad) {
	Mat tmp = Mat::zeros(im.size[0] + 2 * yPad, im.size[1] + 2 * xPad, im.type());
	tmp(Rect(xPad, yPad, im.size[1], im.size[0])) += im;
	out = tmp;
}

void binImToSamples(const Mat &im, Mat &out) {
	assert(im.type() == CV_8UC1);

	out.create(0, 2, CV_32FC1);

	Mat tmp(1, 2, CV_32FC1);
	for (int x = 0; x < im.size[1]; x++) {
		for (int y = 0; y < im.size[0]; y++) {
			if (im.at<uchar>(y, x)) {
				tmp.at<float>(0, 0) = (float)x;
				tmp.at<float>(0, 1) = (float)y;
				out.push_back(tmp);
			}
		}
	}
}

void getCoreTlPoints(const Mat &im, const Size &winSz, vector<PxPoint> &out, int numCores) {
	Mat binIm;
	threshold(im, binIm, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("i", binIm);
	//waitForKeypress();

	pad(binIm, binIm, winSz.width, winSz.height);

	Mat sampMat;
	binImToSamples(binIm, sampMat);

	printf("%d %d\n", sampMat.size[1], sampMat.size[0]);

	Mat labels;
	Mat centers;
	kmeans(sampMat, numCores*2, labels, TermCriteria(TermCriteria::COUNT, 1000, 0), 10, KMEANS_PP_CENTERS, centers);
	printf("kmeans done\n");

	set<PxPoint> tlPts;
	for (int i = 0; i < centers.size[0]; i++) {
		bool converged;
		Point tl((int)centers.at<float>(i, 0) - winSz.width/2, (int)centers.at<float>(i, 1) - winSz.height/2);
		Point newTl = descend(winSz, binIm, tl, converged, false);
		if (converged) {
			bool placed = false;
			for (int dx = -winSz.width/2; dx < winSz.width/2 && !placed; dx++) {
				for (int dy = -winSz.height/2; dy < winSz.height/2; dy++) {
					PxPoint nearPt = {newTl.x + dx, newTl.y + dy};
					if (tlPts.count(nearPt)) {
						placed = true;
						break;
					}
				}
			}
			if (!placed) {
				PxPoint pt = {newTl.x, newTl.y};
				tlPts.emplace(pt);
			}
		}
	}
	printf("center of mass done\n");

	out.clear();
	for (auto pt : tlPts) {
		PxPoint newPt = {pt[0] - winSz.width, pt[1] - winSz.height};
		out.push_back(newPt);
	}
}

void PxPointToSamples(vector<PxPoint> &tlPts, Mat &out, int dim) {
	assert(dim == 0 || dim == 1);
	int size = tlPts.size();
	out.create(size, 1, CV_32FC1);

	for (int i = 0; i < size; i++)
		out.at<float>(i, 0) = (float)tlPts[i][dim];
}

void centerOrdering(Mat &centerMat, Mat &orderMat) {
	assert(centerMat.size().width == 1);
	orderMat.create(centerMat.size().height, centerMat.size().width, CV_32SC1);
	int minIdx = 0, count = 0;
	float min, curMin = -FLT_MAX;
	while (count < centerMat.size().height) {
		min = FLT_MAX;
		for (int i = 0; i < centerMat.size().height; i++) {
			if (centerMat.at<float>(i, 0) <= curMin) {
				continue;
			}
			else if (centerMat.at<float>(i, 0) < min) {
				min = centerMat.at<float>(i, 0);
				minIdx = i;
			}
		}
		orderMat.at<int>(minIdx, 0) = count;
		curMin = min;
		count++;
	}
}

bool findOutliers(int K, vector<PxPoint> &tlPts, const Mat &label)
{
	vector<int> count(K, 0);
	for (int i = 0; i < label.rows; i++)
		count[label.at<int>(i, 0)]++;
	double mean = accumulate(count.begin(), count.end(), 0.0) / K;
	for (int j = 0; j < K; j++)
	{
		if (count[j] < mean / 2)
		{
			printf("detect outliers with label %d\n", j);
			int numRemoved = 0;
			for (int k = 0; k < tlPts.size(); k++)
			{
				if (label.at<int>(k, 0) == j)
				{
					int index = k - numRemoved;
					printf("remove (%d, %d)\n", tlPts[index][0], tlPts[index][1]);
					tlPts.erase(tlPts.begin() + index);
					numRemoved++;
				}
			}
			return true;
		}
	}
	return false;
}

void coreOrdering(vector<PxPoint> &tlPts, vector<CorePt> &corePts, int coreW, int coreH, const string &outdir) {
	Mat xOut, yOut, labelMatX, labelMatY, centerMatX, centerMatY, orderMatX, orderMatY;

	do {
		PxPointToSamples(tlPts, xOut, 0);
		kmeans(xOut, coreW, labelMatX, TermCriteria(TermCriteria::COUNT, 1000, 0), 10, KMEANS_PP_CENTERS, centerMatX);
	} while (findOutliers(coreW, tlPts, labelMatX));
	do {
		PxPointToSamples(tlPts, yOut, 1);
		kmeans(yOut, coreH, labelMatY, TermCriteria(TermCriteria::COUNT, 1000, 0), 10, KMEANS_PP_CENTERS, centerMatY);
	} while (findOutliers(coreH, tlPts, labelMatY));
	PxPointToSamples(tlPts, xOut, 0);
	kmeans(xOut, coreW, labelMatX, TermCriteria(TermCriteria::COUNT, 1000, 0), 10, KMEANS_PP_CENTERS, centerMatX);

	centerOrdering(centerMatX, orderMatX);
	centerOrdering(centerMatY, orderMatY);

	for (int i = 0; i < tlPts.size(); i++) {
		int labelX = labelMatX.at<int>(i, 0);
		int coorX = orderMatX.at<int>(labelX, 0);
		int labelY = labelMatY.at<int>(i, 0);
		int coorY = orderMatY.at<int>(labelY, 0);
		CorePt pt = {coorX, coorY};
		corePts.push_back(pt);
	}

	FILE *fx = fopenWrap((outdir + "/xOut.txt").c_str(), "w");
	FILE *fy = fopenWrap((outdir + "/yOut.txt").c_str(), "w");
	for (int i = 0; i < xOut.rows; i++)
	{
		fprintf(fx, "%f\n", xOut.at<float>(i, 0));
		fprintf(fy, "%f\n", yOut.at<float>(i, 0));
	}
	fclose(fx);
	fclose(fy);
}

void paramParser(const char *argv[],
	string &inDir, string &outDir, string &projName, int &zoomLvls,
	int &gridW, int &gridH, int &coreW, int &coreH, int &winW, int &winH,
	bool &crop) {
	inDir = string(argv[1]);
	outDir = string(argv[2]);
	projName = string(argv[3]);
	zoomLvls = atoi(argv[4]);
	gridW = atoi(argv[5]);
	gridH = atoi(argv[6]);
	coreW = atoi(argv[7]);
	coreH = atoi(argv[8]);
	winW = atoi(argv[9]);
	winH = atoi(argv[10]);
	crop = (atoi(argv[11]) == 1) ? true : false;

	printf("*************** configuration ***************\n");
	printf("inDir: %s\n", inDir.c_str());
	printf("outDir: %s\n", outDir.c_str());
	printf("projName: %s\n", projName.c_str());
	printf("zoomLvls: %d\n", zoomLvls);
	printf("gridW: %d, gridH: %d\n", gridW, gridH);
	printf("coreW: %d, coreH: %d\n", coreW, coreH);
	printf("winW: %d, winH: %d\n", winW, winH);
	printf("crop: %s\n", crop ? "yes" : "no");
	printf("*********************************************\n");
}

int main(int argc, const char *argv[]) {
	if (argc != 12) {
		printf("wrong number of arguments: argc = %d\n", argc);
		exit(-2);
	}

	string inDir, outDir, projName;
	int zoomLvls, gridW, gridH, coreW, coreH, winW, winH;
	bool crop;

	paramParser(argv,
		inDir, outDir, projName, zoomLvls,
		gridW, gridH, coreW, coreH, winW, winH,
		crop);

	Size tileSz(4096, 4096);
	Size gridDim(gridW, gridH);

	if (_mkdir(outDir.c_str()) == -1)
		printf("Folder for cores already created.\n");

	Mat im = imread(inDir + "/cropped.jpg", IMREAD_UNCHANGED);
	printf("%d\n", im.type());
	if (!im.data) {
		printf("Can't read file: %s\n", (inDir + "/cropped.jpg").c_str());
		exit(1);
	}

	Mat im_select = im.clone();
	//Rect roi1(660, 0, im.size().width - 660, 100);
	//im_select(roi1) = 64;
	Mat im_rgb(im.size().height, im.size().width, CV_8UC3);

	//saveDownSampleIm(path, tileSz, gridDim, 5, im);
	//im.convertTo(im, CV_8U, 255);

	Size winSz(winW, winH);
	vector<PxPoint> tlPts;
	vector<CorePt> corePts;
	getCoreTlPoints(im_select, winSz, tlPts, coreW*coreH);
	coreOrdering(tlPts, corePts, coreW, coreH, outDir);

	Laplacian(im, im_rgb, CV_8U, 3, 1, 0);
	applyColorMap(im_rgb, im_rgb, COLORMAP_OCEAN);

	for (int i = 0; i < tlPts.size(); i++) {
		rectangle(im_rgb, Rect(tlPts[i][0], tlPts[i][1], winSz.width, winSz.height), Scalar(0, 0, 255), 2);
		rectangle(im, Rect(tlPts[i][0], tlPts[i][1], winSz.width, winSz.height), Scalar(255, 255, 255), 2);
		ostringstream ss;
		ss << (char)(65 + corePts[i][0] % 26) << corePts[i][1];

		// check duplicate labels
		int duplicate = 0;
		for (int j = 0; j < i; j++) {
			if (corePts[i] == corePts[j])
				duplicate++;
		}
		if (duplicate > 0)
			ss << "(" << duplicate << ")";

		string text = ss.str();
		int fontFace = FONT_HERSHEY_PLAIN;
		double fontScale = 4.5 / text.length();
		int thickness = 2;
		int baseLine = 0;
		Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseLine);
		Point blPt(tlPts[i][0] + (winW - textSize.width) / 2, tlPts[i][1] + (winH + textSize.height) / 2);
		putText(im_rgb, text, blPt, fontFace, fontScale, Scalar(0, 255, 255), thickness, 8);
		putText(im, text, blPt, fontFace, fontScale, Scalar(255, 255, 255), thickness, 8);

		if (crop)
			imCrop(outDir, inDir, Rect(tlPts[i][0] * pow(2, zoomLvls), tlPts[i][1] * pow(2, zoomLvls),
				winSz.width*pow(2, zoomLvls), winSz.height*pow(2, zoomLvls)), tileSz, gridDim, text);
	}

	imwrite(outDir + "/" + projName + "-core.jpg", im);
	imwrite(outDir + "/" + projName + "-core-color.jpg", im_rgb);

	imshow("i", im_rgb);
	waitKey(0);
	return 0;
}
