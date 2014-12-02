#include <cassert>
#include <string>
#include <sstream>
#include <cstdio>
#include <thread>
#include "fetcher.hpp"

//unsigned getImgI(ImgPattern pat, GridPt pt, const Size &szInIms) {
//	assert(pt[0] < szInIms.width && pt[1] < szInIms.height);
//
//	unsigned i;
//	switch (pat) {
//	case ImgPattern::SNAKE_BY_COL:
//		i = pt[0] * szInIms.height;
//		if (pt[0] % 2) {
//			i += szInIms.height - pt[1] - 1;
//		} else {
//			i += pt[1];
//		}
//		break;
//	case ImgPattern::SNAKE_BY_ROW:
//		i = pt[1] * szInIms.width;
//		if (pt[1] % 2) {
//			i += szInIms.width - pt[0] - 1;
//		} else {
//			i += pt[0];
//		}
//		break;
//	default:
//		printf("invalid ImgPattern\n");
//		exit(1);
//	}
//	return i;
//}

static void doFixNaNs(Mat &m) {
	assert(m.type() == CV_32FC1);

	for (int i = 0; i < m.size[0]; i++) {
		for (int j = 0; j < m.size[1]; j++) {
			//if (m.at<float>(i, j) != m.at<float>(i, j)) 
			if (isnan( m.at<float>(i, j)))
			{
				m.at<float>(i, j) = 0;
			}
		}
	}
}

ImgFetcher::ImgFetcher(const vector<string> &imgPaths, Size szInIms, size_t cacheSz, bool fixNaNs) :
	imgPaths(imgPaths),
	szInIms(szInIms),
	row_major((szInIms.width <= szInIms.height) ? true : false),
	cap(MIN(szInIms.width, szInIms.height) + 2),
	imSz(),
	useFastRead(false),
	fixNaNs(fixNaNs),
	cache(cacheSz) {
		if (cacheSz == 0) {
			fprintf(stderr, "err: cache size must be >=1\n");
			exit(1);
		}
}

void ImgFetcher::getMat(const string &file, Mat &out) {
	//printf("reading %s\n", file.c_str());//too much noise?

	cache.get(file, out, [&](const string &file, Mat &out) {
		if (imSz.width == 0 && imSz.height == 0) { 
			//first call, maybe use a static?

			// get the tile size when loading the image for the first time
			out = imread(file, cv::IMREAD_ANYDEPTH);//
			//out = cv::imread()
			if (fixNaNs) {
				doFixNaNs(out);
			}
			imSz = out.size();

			Mat fastOut;
			loadFloatTiffFast(file, fastOut, imSz.width, imSz.height);
			if (fixNaNs) {
				doFixNaNs(fastOut);
			}

			if (countNonZero(out != fastOut) == 0) {
				useFastRead = true;
			} else {
				printf("WARNING: fast read doesn't work for this dataset, so disabling it. Reading might be slow.\n");
			}
		} else {
			if (useFastRead) {
				loadFloatTiffFast(file, out, imSz.width, imSz.height);
			} else {
				loadFloatTiff(file, out, imSz.width, imSz.height);
			}
			if (fixNaNs) {
				doFixNaNs(out);
			}
		}
	});

	if (!out.data) {
		printf("Can't read file: %s\n", file.c_str());
		exit(1);
	}
}

void ImgFetcher::getMat(const GridPt &pt, Mat &out) {
	getMat(imgPaths[pt[1] * szInIms.width + pt[0]], out);
}
