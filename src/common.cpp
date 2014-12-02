#define _CRT_SECURE_NO_WARNINGS

#include "common.hpp"
#include "tiffint.hpp"
#include <cstring>
#include <cmath>
#include <ctime>

void imWriteWin(const string &name, const Mat &im) {
#ifdef IMRW_BUG
	CvMat arr = im;
	cvSaveImage(name.c_str(), (CvArr *)&arr);
#else
	std::vector<int> params;
	params.push_back(IMWRITE_JPEG_PROGRESSIVE);
	//params.push_back(IMWRITE_JPEG_QUALITY);
	//params.push_back(50);
	cv::imwrite(name, im,params);
#endif
}

TransData::TransData() {
}

TransData::TransData(int x, int y, double dist, float conf) :
	trans(x, y),
	dist(dist),
	conf(conf) {
}

// save a CV_32FC1 or CV_64FC1 image
void saveFloatIm(const string &name, const Mat &im) {
	Mat saved;
	normalize(im, saved, 0, 255, NORM_MINMAX);
	saved.convertTo(saved, CV_8U);

	printf("saving %s\n", name.c_str());
	imWriteWin(name, saved);
}

// save and normalize with minPx and maxPx
void saveFloatIm(const string &name, const Mat &im, float minPx, float maxPx) {
	Mat saved = (im - minPx) / (maxPx - minPx) * 255;
	saved.convertTo(saved, CV_8U); // overflow is avoided

	printf("saving %s\n", name.c_str());
	imWriteWin(name, saved);
}

void saveFloatTiff(const string &name, const Mat &im) {
	assert(im.isContinuous());
	saveFloatTiff(name, (float *)im.data, im.size[1], im.size[0]);
}

void loadFloatTiff(const string &name, Mat &im, int w, int h) {
	static float *buf = NULL;
	static int bufLen = 0;

	int newSz = sizeof(float) * w * h;
	if (newSz > bufLen) {
		auto tmp = (float *)realloc(buf, newSz);
		if (tmp == NULL)
		{
			std::cout << "Can't expand buffer, go buy more RAM." << std::endl;
		}
		else
		{
			buf = tmp;
		}
		bufLen = newSz;
	}
	loadFloatTiff(name, buf);
	im = Mat(h, w, CV_32FC1, buf).clone();
}

void loadFloatTiffFast(const string &name, Mat &im, int w, int h) {
	static float *buf = NULL;
	static int bufLen = 0;

	int newSz = sizeof(float) * w * h;
	if (newSz > bufLen) {
		auto tmp = (float *)realloc(buf, newSz);
		if (tmp == NULL)
		{
			std::cout << "Can't expand buffer, go buy more RAM." << std::endl;
		}
		else
		{
			buf = tmp;
		}
		bufLen = newSz;

	}
	loadFloatTiffFast(name, buf, w, h);
	im = Mat(h, w, CV_32FC1, buf).clone();
}

void blit(const Mat &a, Mat &b, Point tl) {
	//	printf("blit tl: %d %d\n", tl.x, tl.y);
	Rect aRoi(0, 0, a.size[1], a.size[0]);
	if (tl.x < 0) {
		aRoi.x = -tl.x;
		aRoi.width += tl.x;
		tl.x = 0;
	}
	if (tl.x + aRoi.width > b.size[1]) {
		aRoi.width = b.size[1] - tl.x;
	}
	if (tl.y < 0) {
		aRoi.y = -tl.y;
		aRoi.height += tl.y;
		tl.y = 0;
	}
	if (tl.y + aRoi.height > b.size[0]) {
		aRoi.height = b.size[0] - tl.y;
	}
	//	cout << Rect(tl, aRoi.size()) << endl;
	//	cout << aRoi << endl;
	b(Rect(tl, aRoi.size())) += a(aRoi);
}

static void loadGridIm(const string &path,const GridPt& pt, Mat &out, int w, int h)
{
	ostringstream ss;
	ss << path << "/" << pt[1] << "_" << pt[0] << "_0.tiff";
	string s = ss.str();

	printf("loading %s\n", s.c_str());
	loadFloatTiff(s, out, w, h);

	if (!out.data)
	{
		printf("Can't read file: %s\n", s.c_str());
		exit(1);
	}
	assert(out.type() == CV_32FC1); 
}

void imCrop(const string &outPath, const string &inPath, const Rect& outRec,const Size& tileSz,const Size& gridDim, const string &fileName)
{
	// check validity
	Point tl = outRec.tl();
	Point br = outRec.br();
	Size sz = outRec.size();

	Size fullSz(tileSz.width * gridDim.width, tileSz.height * gridDim.height);
	printf("tl: %d, %d/br: %d, %d\n", tl.x, tl.y, br.x, br.y);
	assert(br.x > 0 && br.y > 0 && tl.x < fullSz.width && tl.y < fullSz.height);

	Mat outIm(sz, CV_32FC1);
	printf("size of outIm: w: %d, h: %d\n", outIm.size().width, outIm.size().height);
	Point relTL; // relTL is relative to top-left of outIm

	int startX = tl.x / tileSz.width;
	int endX = br.x / tileSz.width;
	int startY = tl.y / tileSz.height;
	int endY = br.y / tileSz.height;

	printf("start: (%d, %d), end: (%d, %d)\n", startX, startY, endX, endY);

	for (int i = startY; i <= endY; i++)
	{
		for (int j = startX; j <= endX; j++)
		{
			if (i < 0 || i >= gridDim.height || j < 0 || j >= gridDim.width)
				continue;
			Mat temp;
			printf("j: %d, i: %d\n", j, i);
			relTL.x = tileSz.width * j - tl.x;
			relTL.y = tileSz.height * i - tl.y;
			printf("relTL.x: %d, relTL.y: %d\n", relTL.x, relTL.y);
			loadGridIm(inPath, makePt(j, i), temp, tileSz.width, tileSz.height); 
			blit(temp, outIm, relTL);
		}
	}

	ostringstream ss;
	ss << outPath << "/" << fileName << ".tiff";
	string s = ss.str();
	saveFloatTiff(s, outIm);
}

void saveDownSampleIm(const string &path,const Size& tileSz,const Size& gridDim, int scaleLevel, Mat &outIm)
{
	Size outTileSz((int)(tileSz.width / pow(2.0, (double)scaleLevel)), (int)(tileSz.height / pow(2.0, (double)scaleLevel)));
	Size outFullSz(gridDim.width * outTileSz.width, gridDim.height * outTileSz.height);
	assert((outFullSz.width < 32768) && (outFullSz.height < 32768));

	printf("outTileSz: (%d, %d)\n", outTileSz.width, outTileSz.height);
	printf("outFullSz: (%d, %d)\n", outFullSz.width, outFullSz.height);

	outIm.create(outFullSz, CV_32FC1);
	Point relTL; // relTL is relative to top-left of outIm

	for (int i = 0; i < gridDim.height; i++)
	{
		for (int j = 0; j < gridDim.width; j++)
		{
			Mat temp;
			loadGridIm(path, makePt(j, i), temp, tileSz.width, tileSz.height);
			resize(temp, temp, Size(0, 0), 1 / pow(2.0, (double)scaleLevel), 1 / pow(2.0, (double)scaleLevel), INTER_NEAREST);
			assert((temp.size().width == outTileSz.width) && (temp.size().height == outTileSz.height));
			relTL.x = j * outTileSz.width;
			relTL.y = i * outTileSz.height;
			blit(temp, outIm, relTL);
		}
	}

	saveFloatTiff("../../../bin/down.tif", outIm);
}

string getCurTime()
{
	time_t now = time(0);
	tm *ltm = localtime(&now);

	ostringstream ss;
	ss << 1 + ltm->tm_hour << ":" << 1 + ltm->tm_min << ":" << 1 + ltm->tm_sec << endl;
	string s = ss.str();

	return s;
}

static string getParentDir(string &path)
{
	string dir = path;
	for (int i = path.size() - 1; i >= 0; i--)// counting down is dangerous if unsinged integer
	{
		if (path[i] == '/' || path[i] == '\\')
			break;
		dir.erase(i, 1);
	}
	return dir;
}

void getBg(Mat &bgIm, vector<string> &imPaths, float bgSub, int w, int h)
{
	if (bgSub == 0.0) // not doing background subtraction
		return;

	int choose = (int)(imPaths.size() * bgSub);

	map<double, int> meanToIndex;
	for (int i = 0; i < imPaths.size(); i++)
	{
		cout << imPaths[i] << endl;
		Mat im;
		loadFloatTiff(imPaths[i], im, w, h);
		double mean = cv::mean(abs(im))[0];
		while (!meanToIndex.insert(pair<double, int>(mean, i)).second)
		{
			// deal with duplicate means
			mean += DBL_MIN;
		}
	}

	Mat bg = Mat::zeros(h, w, CV_32F);
	int count = 0;
	for (auto it = meanToIndex.begin(); it != meanToIndex.end(); ++it)
	{
		if (count >= choose) break;
		std::cout << it->first << " => " << it->second << '\n';
		Mat im;
		loadFloatTiff(imPaths[it->second], im, w, h);
		bg += im;
		count++;
	}

	bg /= choose;

	ostringstream ss;
	ss << getParentDir(imPaths[0]) << "bgimage.tif";
	saveFloatTiff(ss.str(), bg);

	bgIm = bg.clone();
}

FILE *fopenWrap(const char *path, const char *mode) {
	FILE *f = fopen(path, mode);
	if (!f) {
		printf("err: unable to open file %s for %s\n",
			path,
			mode[0] == 'w' ? "writing" : (mode[0] == 'a' ? "appending" : "reading")
			);
		exit(1);
	}
	return f;
}