#include "translate.hpp"
#include <future>
#include <thread>
#include <utility>

int peakRadius = -1;

typedef array<int, 2> GridPtOff;	// grid point offset

static GridPtOff makeOff(int a, int b) {
	GridPtOff o = {{a, b}};
	return o;
}

static int fround(float f) {
	return (int)floor(f + 0.5);
}

// since fftw is not thread safe according to the site 
static void *fftwf_malloc_thr(size_t sz) {
	static mutex m;
	m.lock();
	void *r = fftwf_malloc(sz);
	m.unlock();
	return r;
}

static void fftwf_free_thr(void *p) {
	static mutex m;
	m.lock();
	fftwf_free(p);
	m.unlock();
}

// store the map from grid point offset to mask used in phase correlation peak finding
static void storeHintToMask(map<GridPtOff, Mat> &hintToMask, const Size &imSz, const Point2f &absHint, const MaxDists &dists) {
	for (int xOff = -1; xOff <= 1; xOff++) {
		for (int yOff = -1; yOff <= 1; yOff++) {
			if (!xOff && !yOff) {
				continue;
			}

			int maxDist;
			if (xOff) {
				if (yOff) {
					maxDist = dists.xy;
				} else {
					maxDist = dists.x;
				}
			} else {
				maxDist = dists.y;
			}

			Mat mask = Mat::zeros(Size(imSz.width + 2, imSz.height), CV_8UC1);

			Point baseXY(fround(absHint.x * xOff), fround(absHint.y * yOff));
			if (baseXY.x > imSz.width) {
				baseXY.x -= imSz.width;
			} else if (baseXY.x < 0) {
				baseXY.x += imSz.width;
			}
			if (baseXY.y > imSz.height) {
				baseXY.y -= imSz.height;
			} else if (baseXY.y < 0) {
				baseXY.y += imSz.height;
			}

			if (xOff) {
				if (yOff) {
					circle(mask, baseXY, maxDist, Scalar(255, 255, 255, 255), -1);
				} else {
					for (int y = baseXY.y - imSz.height; y <= baseXY.y + imSz.height; y += imSz.height) {
						circle(mask, Point(baseXY.x, y), maxDist, Scalar(255, 255, 255, 255), -1);
					}
				}
			} else {
				for (int x = baseXY.x - imSz.width; x <= baseXY.x + imSz.width; x += imSz.width) {
					circle(mask, Point(x, baseXY.y), maxDist, Scalar(255, 255, 255, 255), -1);
				}
			}

			//			Mat tmp;
			//			resize(mask, tmp, Size(), 0.4, 0.4);
			//			imshow("m", tmp);
			//			waitKey(0);

			hintToMask.emplace(makeOff(xOff, yOff), mask);
		}
	}
}

// this struct is not really needed anymore but kept to make it easy to extend. it was used to store least squares
// confidence using the image data itself instead of the result of phase correlation.
struct FFTHolder {
	float *fft;
	float preConfs[9];

	FFTHolder() : fft(NULL) {
	}

	FFTHolder(const Mat &im, const Point2f &absHint, const fftwf_plan &plan) {
		fft = getFFT(im.data, im.size(), plan);
	}
};

float getConfSumNbrs(const Mat &peakIm, const Point &maxLoc, double bestSqDist) {
	assert(peakRadius >= 0);
	assert(peakIm.type() == CV_32FC1);

	float s = 0;
	for (int x = maxLoc.x - peakRadius; x <= maxLoc.x + peakRadius; x++) {
		for (int y = maxLoc.y - peakRadius; y <= maxLoc.y + peakRadius; y++) {
			int usedX = x, usedY = y;

			if (usedX < 0) {
				usedX += peakIm.size[1];
			} else if (usedX >= peakIm.size[1]) {
				usedX -= peakIm.size[1];
			}

			if (usedY < 0) {
				usedY += peakIm.size[0];
			} else if (usedY >= peakIm.size[0]) {
				usedY -= peakIm.size[0];
			}

			s += peakIm.at<float>(usedY, usedX);
		}
	}

	s /= 1000;

	if (s <= 0) {
		s = static_cast<decltype(s)>(1e-10);	// some small value
	}

	return s;
}

float getConfInvSqDist(const Mat &peakIm, const Point &maxLoc, double bestSqDist) {
	return (float)(1000 / bestSqDist);
}

static TransData phaseCorrThr(const shared_future<FFTHolder> &fixFut,  const shared_future<FFTHolder> &nbrFut, const fftwf_plan &c2rPlan, const PtPair &pair,
							  const Point2f &absHint, const map<GridPtOff, Mat> &hintToMask, const Size &imSz) {
	//printf("phase %d %d, %d %d\n", pair[0][0], pair[0][1], pair[1][0], pair[1][1]);
	float *buf = (float *)fftwf_malloc_thr(sizeof(float) * getFFTLen(imSz));

	GridPtOff off = {{pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]}};	// vector from fix to nbr
	const Mat &mask = hintToMask.at(off);

	Point2f hint = absHint;
	hint.x *= off[0];
	hint.y *= off[1];

	FFTHolder fixFFT = fixFut.get();
	FFTHolder nbrFFT = nbrFut.get();

	/* testing
	Mat fix(imSz.height, imSz.width + 2, CV_32F, fixFFT.fft);
	Mat nbr(imSz.height, imSz.width + 2, CV_32F, nbrFFT.fft);
	double minVal, maxVal;
	Point minLoc2, maxLoc2;
	minMaxLoc(fix, &minVal, &maxVal, &minLoc2, &maxLoc2);
	printf("min: %lf, (%d, %d)\n", minVal, minLoc2.x, minLoc2.y);
	printf("max: %lf, (%d, %d)\n", maxVal, maxLoc2.x, maxLoc2.y);
	minMaxLoc(nbr, &minVal, &maxVal, &minLoc2, &maxLoc2);
	printf("min: %lf, (%d, %d)\n", minVal, minLoc2.x, minLoc2.y);
	printf("mmaxin: %lf, (%d, %d)\n", maxVal, maxLoc2.x, maxLoc2.y);
	imshow("a", fix);
	waitKey(0);
	*/

	double dist;
	float conf;
	Point maxLoc;
	Point bestPt = phaseCorr(fixFFT.fft, nbrFFT.fft, imSz, c2rPlan, buf, hint, dist, mask, maxLoc, conf, getConfSumNbrs);

	fftwf_free_thr(buf);

	//	printf("got conf %f old conf %f rat %f, maxLoc %d %d phase %d %d, %d %d, dist %lf, mask sz: %d %d\n\n", conf, oldConf, conf / 1000 / oldConf, maxLoc.x, maxLoc.y,
	//			pair[0][0], pair[0][1], pair[1][0], pair[1][1], dist, mask.size[1], mask.size[0]);
	TransData dat(bestPt.x, bestPt.y, dist, conf);
	//printf("done phase %d %d, %d %d\n", pair[0][0], pair[0][1], pair[1][0], pair[1][1]);

	return dat;
}

// return false if the gridPt is not in grid
static inline bool ptInGrid(const GridPt& pt, ImgFetcher &fetcher)
{
	return !((pt[0] < 0) ||
		(pt[0] >= fetcher.szInIms.width) ||
		(pt[1] < 0) ||
		(pt[1] >= fetcher.szInIms.height));
}

// return false when it gets out of range
static bool nextCoor(GridPt &coor, ImgFetcher &fetcher)
{
	if (fetcher.row_major)
	{
		coor[0]++;
		if (coor[0] == fetcher.szInIms.width)
		{
			coor[0] = 0;
			coor[1]++;
		}
	}
	else // col_major
	{
		coor[1]++;
		if (coor[1] == fetcher.szInIms.height)
		{
			coor[1] = 0;
			coor[0]++;
		}
	}
	if (!ptInGrid(coor, fetcher)) return false;
	return true;
}

float *getFFT(const void *matData, const Size &imSz, const fftwf_plan &plan) {
	float *fft = (float *)fftwf_malloc_thr(sizeof(float) * getFFTLen(imSz));
	if (!fft) {
		throw bad_alloc();
	}

	int inI = 0, outI = 0;
	for (int r = 0; r < imSz.height; r++) {
		for (int c = 0; c < imSz.width; c++) {
			fft[outI++] = ((const float *)matData)[inI++];
			if (((const float *)matData)[inI - 1] != ((const float *)matData)[inI - 1]) {
				//printf(" UH UH %d %d\n", r, c);
			}
		}
		fft[outI++] = 0;	// zero-pad twice
		fft[outI++] = 0;
	}

	fftwf_execute_dft_r2c(plan, fft, (fftwf_complex *)fft);

	// testing
//	static mutex mut;
//	mut.lock();
//	Mat im(imSz, CV_32FC1, (void *)matData);
//
//	double minVal, maxVal;
//	Point minLoc, maxLoc;
//
//	minMaxLoc(im, &minVal, &maxVal, &minLoc, &maxLoc);
//	printf("min: %lf, (%d, %d)\n", minVal, minLoc.x, minLoc.y);
//	printf("max: %lf, (%d, %d)\n", maxVal, maxLoc.x, maxLoc.y);
//	imshow("hn", im);
//	waitKey(0);
//	mut.unlock();
//
//	Mat matDatam(imSz.height, imSz.width+2, CV_32F, (float *)fft);
//	minMaxLoc(matDatam, &minVal, &maxVal, &minLoc, &maxLoc);
//	printf("min: %lf, (%d, %d)\n", minVal, minLoc.x, minLoc.y);
//	printf("max: %lf, (%d, %d)\n", maxVal, maxLoc.x, maxLoc.y);
//
//	imshow("immmmm", matDatam);
//	waitKey(0);
	//

	return fft;
}

// buf must be fftwf_malloc'd with size `sizeof(float) * fftLen`
//
// im should be approx offset from fix by hint.
Point phaseCorr(const float *fix, const float *im, const Size &imSz, const fftwf_plan &plan, float *buf, const Point2f &hint,
				double &bestSqDist, const Mat &mask, Point &maxLoc, float &conf, const ConfGetter getConf, const char *saveIm) {
	unsigned fftLen = getFFTLen(imSz);
	for (unsigned i = 0; i < fftLen; i += 2) {
		float a = fix[i] * im[i] + fix[i + 1] * im[i + 1];
		float b = fix[i + 1] * im[i] - fix[i] * im[i + 1];
		float norm = sqrt(a * a + b * b);
		buf[i] = a / norm;
		buf[i + 1] = b / norm;
	}

	fftwf_execute_dft_c2r(plan, (fftwf_complex *)buf, buf);

	Mat bufMat(imSz.height, imSz.width + 2, CV_32FC1, buf);
	//	bufMat = abs(bufMat);
	blur(bufMat, bufMat, Size(21, 21));

	if (saveIm) {
		saveFloatIm(saveIm, bufMat);
	}

	minMaxLoc(bufMat, NULL, NULL, NULL, &maxLoc, mask);

	// there are four potential shifts corresponding to one peak
	// we choose the shift that is closest to the microscope's guess for the offset between the two
	Point bestPt;
	bestSqDist = 1e99;
	for (int dx = -imSz.width; dx <= 0; dx += imSz.width) {
		for (int dy = -imSz.height; dy <= 0; dy += imSz.height) {
			Point curPt(maxLoc.x + dx, maxLoc.y + dy);
			double curSqDist = getSqDist(curPt, hint);
			if (curSqDist < bestSqDist) {
				bestSqDist = curSqDist;
				bestPt = curPt;
			}
		}
	}

	conf = getConf(bufMat, maxLoc, bestSqDist);

	return bestPt;
}

// store translations into transMap
void storeTrans(ImgFetcher &fetcher, const Point2f &absHint, PairToTransData &transMap, const MaxDists &dists) {
	vector<GridPtOff> imOffs;
	if (fetcher.row_major) {
		imOffs.push_back(makeOff(-1, 0));
		imOffs.push_back(makeOff(-1, -1));
		imOffs.push_back(makeOff(0, -1));
		imOffs.push_back(makeOff(1, -1));
	} else {
		imOffs.push_back(makeOff(0, -1));
		imOffs.push_back(makeOff(-1, -1));
		imOffs.push_back(makeOff(-1, 0));
		imOffs.push_back(makeOff(-1, 1));
	}

	map<PtPair, shared_future<TransData>> pairToTransFut;
	map<GridPt, shared_future<FFTHolder>> ptToFFTFut;

	unsigned loaded = 0;
	GridPt fixPt = {{0, 0}};
	GridPt waitPt = {{0, 0}};
	Mat cur;

	fetcher.getMat(fixPt, cur);
	Size imSz = cur.size();
	unsigned fftLen = getFFTLen(imSz);

	map<GridPtOff, Mat> hintToMask;
	storeHintToMask(hintToMask, imSz, absHint, dists);

	float *tmp = (float *)fftwf_malloc_thr(sizeof(float) * fftLen);
	fftwf_plan r2cPlan = fftwf_plan_dft_r2c_2d(imSz.height, imSz.width, tmp, (fftwf_complex *)tmp, FFTW_MEASURE);
	fftwf_plan c2rPlan = fftwf_plan_dft_c2r_2d(imSz.height, imSz.width, (fftwf_complex *)tmp, tmp, FFTW_MEASURE);
	fftwf_free_thr(tmp);

	bool readDone = false;
	while (true) {
		//a dirty kind of event loop
		if (loaded > fetcher.cap || readDone) {
			//			printf("start free waitPt %d %d\n", waitPt[0], waitPt[1]);
			// free oldest image, at waitPt
			for (auto &off: imOffs) {
				// *subtract* offset to avoid duplicating pairs
				GridPt nbrPt = {{waitPt[0] - off[0], waitPt[1] - off[1]}};
				if (ptInGrid(nbrPt, fetcher)) {
					PtPair pair = {{waitPt, nbrPt}};
					shared_future<TransData> transFut;
					if (!lookupPair(pairToTransFut, pair, transFut)) {
						printf("err: future of pair %d %d to %d %d not found\n", pair[0][0], pair[0][1], pair[1][0], pair[1][1]);
						exit(1);
					}
					transMap.emplace(pair, transFut.get());
					pairToTransFut.erase(pair);
				}
			}
			fftwf_free_thr(ptToFFTFut[waitPt].get().fft);
			ptToFFTFut.erase(waitPt);

			if (!nextCoor(waitPt, fetcher)) {
				break;
			}
			loaded--;
		}

		if (!readDone) {
			//printf("emplace fft at %d %d\n", fixPt[0], fixPt[1]);
			fetcher.getMat(fixPt, cur);

			// fft only supports 32-bit float with even width, for now
			assert(cur.type() == CV_32FC1 && (int)cur.step[0] == cur.size().width * 4 && cur.step[1] == 4 && cur.size().width % 2 == 0);
			assert(cur.isContinuous());

			ptToFFTFut.emplace(fixPt, async(launch::async,
				[&r2cPlan, &absHint](Mat im) {
					return FFTHolder(im, absHint, r2cPlan);
			},
				cur
				));

			for (auto &off: imOffs) {
				GridPt nbrPt = {{fixPt[0] + off[0], fixPt[1] + off[1]}};
				if (ptInGrid(nbrPt, fetcher)) {
					PtPair pair = {{fixPt, nbrPt}};
					//					printf("emplace pair transfut %d %d, %d %d\n", pair[0][0], pair[0][1], pair[1][0], pair[1][1]);

					// needed since VS2012 async() can't take functions with too many arguments :(
					shared_future<FFTHolder> &a = ptToFFTFut[fixPt];
					shared_future<FFTHolder> &b = ptToFFTFut[nbrPt];
					pairToTransFut.emplace(pair, async(launch::async, [=] {
						return phaseCorrThr(a, b, c2rPlan, pair, absHint, hintToMask, imSz);
					}));
				}
			}

			loaded++;
			if (!nextCoor(fixPt, fetcher)) {
				readDone = true;
			}
		}
	}

	fftwf_destroy_plan(r2cPlan);
	fftwf_destroy_plan(c2rPlan);
}

void writeTranMap(const string &file, const PairToTransData &transMap)
{
	const char *fileC = file.c_str();
	printf("writing %s\n", fileC);
	FILE *pFile = fopen(fileC, "w");
	if (!pFile) {
		printf("err: unable to open file %s for writing\n", fileC);
		exit(1);
	}

	for (auto it = transMap.begin(); it != transMap.end(); ++it)
	{
		auto ref = it->second;
		fprintf(pFile, "(%d, %d)->(%d, %d): (%d, %d, %lf, %f)\n",
			it->first[0][0], it->first[0][1], it->first[1][0], it->first[1][1], it->second.trans.x, ref.trans.y, ref.dist, ref.conf);
	}
	fclose(pFile);
	printf("done\n");
}

void readTranMap(const string &file, PairToTransData &transMap) {
	const char *fileC = file.c_str();
	FILE *f = fopen(fileC, "r");
	if (!f) {
		printf("err: unable to open file %s\n", fileC);
		exit(1);
	}

	while (!feof(f)) {
		GridPt p1, p2;
		Point trans;
		double dist;
		float conf;
		if (fscanf(f, "(%u, %u)->(%u, %u): (%d, %d, %lf, %f)\n",
			&p1[0], &p1[1], &p2[0], &p2[1], &trans.x, &trans.y,
			&dist, &conf) == EOF) return;
		PtPair pair = {{p1, p2}};
		TransData dat(trans.x, trans.y, dist, conf);
		transMap.emplace(pair, dat);
	}
	printf("%lu pairs read from %s\n", transMap.size(), fileC);
}

void rmOutliers(PairToTransData &transMap) {
	vector<pair<double, PtPair>> distAndPairs(transMap.size());
	int i = 0;
	for (auto it = transMap.begin(); it != transMap.end(); ++it) {
		distAndPairs[i++] = pair<double, PtPair>(it->second.dist, it->first);
	}

	sort(distAndPairs.begin(), distAndPairs.end());

	double Q1 = distAndPairs[(int)(distAndPairs.size() * 0.25)].first;
	double Q3 = distAndPairs[(int)(distAndPairs.size() * 0.75)].first;
	printf("Q1 Q3 %f %f\n", Q1, Q3);
	double iqr = Q3 - Q1;
	double high = Q3 + iqr;

	for (auto it = distAndPairs.begin(); it != distAndPairs.end(); ++it) {
		if (it->first > high) {
			transMap[it->second].conf = 0;
			printf("erase (%u %u) (%u %u)\n", it->second[0][0], it->second[0][1], it->second[1][0], it->second[1][1]);
		}
	}
}
