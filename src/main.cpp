#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <fftw3.h>
#include <string>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <regex>
//
#include "fetcher.hpp"
#include "optimize.hpp"
#include "stitch.hpp"
#include "common.hpp"
#include "translate.hpp"
#include "timeslice.hpp"


#define MINVEC(v) *min_element(v.begin(), v.end())
#define MAXVEC(v) *max_element(v.begin(), v.end())

#define MINMAX_PROP 0.3f

#define ABORT(msg) do { \
	fprintf(stderr, "err: " msg "\n"); \
	exit(1); \
} while (0)

#define STR(a) #a
#define R(var, re)  static char var##_[] = STR(re);\
 const char * var = ( var##_[ sizeof(var##_) - 2] = '\0',  (var##_ + 1) );

using namespace std;
void checkforfile(const string& name) // throws IOException() just like teh java
{
	boost::filesystem::path pathway(name);
	auto foo = boost::filesystem::canonical(pathway).make_preferred();
	if (!boost::filesystem::exists(foo))
	{
		auto errormsg = string("Can't find file") + name;
		throw std::runtime_error(errormsg.c_str());
	}
}

class UserInput
{
	static string setFile(const char* in)
	{//do some checking with boost::filesystem
		auto name = string(in);
		checkforfile(name);
		return name;
	}
	static string setDir(const char* in)
	{//do some checking with boost::filesystem
		auto name = string(in);
		if (!boost::filesystem::exists(name))
		{
			auto madeit = boost::filesystem::create_directory(name);
			if (madeit==false)//made a directory but really didn't?
			{
				boost::filesystem::path full_path(boost::filesystem::current_path());
				std::string errormsg = string("Can't find folder: ") + name + string(" ") + full_path.string();
				throw std::runtime_error(errormsg.c_str());
			}
			else
			{
				std::cout << "Made a folder called " << name << std::endl;
			}
		}
		return string(in);
	}
	static float getOffset(const char* in)
	{
		return static_cast<float>(atof(in));
	}
	static cv::Size getSize(const char* cols, const char* rows)
	{
		Size szInIms(atoi(cols), atoi(rows)); // grid size of input images
		return szInIms;
	}
	static int getCacheSize(const char* in)
	{
		return atoi(in);
	}
	static MaxDists getMaxDists(const char* x, const char* y, const char* diag)
	{
		return{ atoi(x), atoi(y), atoi(diag) };
	}
	static float getPower(const char* in)
	{
		auto val = static_cast<float>(atof(in));
		if (val < 0)
		{
			std::string errormsg = string("Peak power should be non-negative:") + in;
			throw std::runtime_error(errormsg.c_str());
		}
		return val;
	}
	static int getPeakRadius(const char* in)
	{
		auto val = static_cast<int>(atoi(in));
		if (val < 0)
		{
			std::string errormsg = string("Peak radius should be non-negative:") + in;
			throw std::runtime_error(errormsg.c_str());
		}
		return val;
	}
	static bool getFixNans(const char* in)
	{
		return atoi(in) ? true : false;
	}
	static float getBGSubtract(const char* in)
	{
		float val;
		auto betterbeone = sscanf(in, "%f", &val);
		if (betterbeone != 1)
		{
			std::string errormsg = string("Could not read bg value, saw:") + in;
			throw std::runtime_error(errormsg.c_str());
		}
		return (val >= 1 || val < 0) ? 0.0f : val;
	}
	void fixup()
	{
		if (xOff > imSz.width || yOff > imSz.height) {
			printf("*** WARNING: xOff or yOff greater than image size (no overlap) ***\n");
			printf("*** Setting overlap to be 1 pixel ***\n");
			if (xOff > imSz.width) {
				xOff = (float)(imSz.width - 1);
			}
			if (yOff > imSz.height) {
				yOff = (float)(imSz.height - 1);
			}
		}
	}
	void argcheck(int argc)
	{

		if (argc != 20)
		{
			std::string errormsg = string("Wrong number of arguments saw: ") + std::to_string(argc);
			throw std::runtime_error(errormsg.c_str());
		}
	}
public:
	string outDir, poslist;
	bool usePoslist;
	float xOff, yOff;
	cv::Size szInIms, imSz, tileSz;
	int cacheSz;
	MaxDists dists;
	float weightPwr;
	float bgSub;
	bool fixNaNs;
	UserInput(int argc, const char *argv[])
	{
		try
		{
			argcheck(argc);
			poslist = setFile(argv[1]);
			outDir = setDir(argv[2]);
			usePoslist = !strcmp(argv[13], "1");
			xOff = getOffset(argv[3]);
			yOff = getOffset(argv[4]);
			szInIms = getSize(argv[5], argv[6]);
			cacheSz = getCacheSize(argv[10]);
			dists = getMaxDists(argv[7], argv[8], argv[9]);
			imSz = getSize(argv[11], argv[12]);
			weightPwr = getPower(argv[14]);
			peakRadius = getPeakRadius(argv[15]);//conceptually flawed because the bluring does this
			fixNaNs = getFixNans(argv[16]);
			bgSub = getBGSubtract(argv[17]);
			tileSz = getSize(argv[18], argv[19]);
		}
		catch (std::exception &e)
		{
			std::cout << "Invalid Arguments:" << e.what() << std::endl;
			for (int i = 0; i < argc; i++)
			{
				std::cout << i << ":" << argv[i] << std::endl;
			}
			std::cout << "Do you have spaces in your file names?" << std::endl;
			std::cout << std::endl;
			exit(20);
		}
		//sanity checks, moved
		fixup();
	}
	friend ostream& operator<<(ostream& os, const UserInput& a)
	{
		os << STR(a.outDir) << ":" << a.outDir << std::endl;
		os << STR(a.poslist) << ":" << a.poslist << std::endl;
		os << STR(a.xOff) << ":" << a.xOff << std::endl;
		os << STR(a.yOff) << ":" << a.yOff << std::endl;
		os << STR(a.szInIms) << ":" << a.szInIms << std::endl;
		os << STR(a.cacheSz) << ":" << a.cacheSz << std::endl;
		os << STR(a.dists) << ":" << a.dists << std::endl;
		os << STR(a.weightPwr) << ":" << a.weightPwr << std::endl;
		os << STR(a.bgSub) << ":" << a.bgSub << std::endl;
		os << STR(a.fixNaNs) << ":" << a.fixNaNs << std::endl;
		os << STR(a.tileSz) << ":" << a.tileSz << std::endl;
		return os;
	}
};

void loadImgData(const UserInput& input, vector<string> &out, LSQRVectors &trans, bool keepall)
{
	TimeSlice t("Stiching List Parsed: ");
	try {
		auto path = input.poslist;
		ifstream f(path);
		if (!f.is_open()) {
			fprintf(stderr, "err: can't open %s for reading\n", path.c_str());
			exit(1);
		}
		out.clear();
		trans.clear();
		string line;
		//http://stackoverflow.com/questions/3978351/how-to-avoid-backslash-escape-when-writing-regular-expression-in-c-c
		R(re, "(.*?.tif)[^0-9\r\n\-\+]+([-+]?[0-9]*\.?[0-9]+)[^0-9\r\n\-\+]+([-+]?[0-9]*\.?[0-9]+)");
		std::regex matchme(re);
		auto isZero = [](float val){ return (val < FLT_EPSILON) && ((val > -FLT_EPSILON)); };
		while (getline(f, line)) {
			std::cmatch res;
			std::regex_search(line.c_str(), res, matchme);
			if (line.empty() || line[0] == ' ' || line[0] == '#')
			{
				continue;//skip
			}
			//
			if (res.size() != 4)
			{
				auto msg = string("Invalid Line: ") + line;
				throw std::runtime_error(msg.c_str());//?
			}
			auto x = stof(res[2]);
			auto y = stof(res[3]);
			auto name = res[1].str();
			checkforfile(name);
			out.push_back(name);
			std::cout << name << "," << x << "," << y << std::endl;
			if ((!isZero(x) || !isZero(y)) || keepall)
			{
				trans.push_back(x, y);
			}
		}
		/*
		assert(givenTrans.size() == imPaths.size() - 1); // You push one for every file so why should they be different????, this could only happen if one position was exactly zero?
		auto one = trans.size();
		auto two = out.size();
		if (one != two - 1)
		{
		throw std::exception("Exactly one point must be on (0,0)");//?
		}//I'm confuzzled by this
		*/
		if (out.size() != input.szInIms.area())
		{
			auto msg = string("Not enough images: expecting ") + std::to_string(input.szInIms.area()) + " but found " + std::to_string(out.size());
			throw std::runtime_error(msg.c_str());//?
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		exit(20);
	}
}

void writePoslist(const string &file, const vector<float> &xs, const vector<float> &ys, const ImgFetcher &fetch) {
	assert(xs.size() == fetch.szInIms.area() && ys.size() == fetch.szInIms.area());

	FILE *f = fopenWrap(file.c_str(), "w");
	for (int i = 0; i < xs.size(); i++) 
	{
		fprintf(f, "%s; ; (%f, %f)\n", fetch.imgPaths[i].c_str(), xs[i], ys[i]);
	}
	fclose(f);
	printf("wrote poslist at: %s\n", file.c_str());
}

void writeWeights(const string &file, const PairToTransData &transMap) {
	FILE *f = fopenWrap(file.c_str(), "w");
	for (auto e : transMap) {
		fprintf(f, "(%d %d)->(%d %d) %e\n", e.first[0][0], e.first[0][1], e.first[1][0], e.first[1][1], e.second.conf);
	}
	fclose(f);
	printf("wrote weights at: %s\n", file.c_str());
}


int main(int argc, const char *argv[])
{
	std::cout << "Starting Stiching..." << std::endl;
	TimeSlice t("Stiching: ");
	// parse arguments
	if (!useOptimized())
	{
		printf("optimizing opencv\n"); // An quality line
		cv::setUseOptimized(true);
	}
	UserInput u(argc, argv);
	std::cout << u << std::endl;
	// parse poslist
	vector<string> imPaths;
	LSQRVectors givenTrans;
	loadImgData(u, imPaths, givenTrans,u.usePoslist);

	auto noppp = givenTrans.size();

	// end argument parsing
	// background subtraction
	Mat bgIm;
	getBg(bgIm, imPaths, u.bgSub, u.imSz.width, u.imSz.height);
	if (!fftwf_init_threads()) {
		printf("error initializing multi-threaded FFTW\n");
		exit(1);
	}
	fftwf_plan_with_nthreads(4);

	ImgFetcher fetch(imPaths, u.szInIms, u.cacheSz, u.fixNaNs);
	LSQRVectors newTrans(fetch.szInIms);
	if (u.usePoslist)
	{
		std::cout << "Not doing aligment!" << std::endl;
		newTrans = givenTrans;
	}
	else
	{
		TimeSlice t1("Phase Correlation");
		// phase correlation
		PairToTransData transMap;

		// uncomment to use stored phase corr results
		//		readTranMap(outDir + "/phasecorr_pairs.txt", transMap);

		// uncomment to run phase corr and store the results
		storeTrans(fetch, Point2f(u.xOff, u.yOff), transMap, u.dists);
		writeTranMap(u.outDir + "/phasecorr_pairs.txt", transMap);

		float maxConf = 0;
		for (auto &e : transMap) {
			e.second.conf = pow(e.second.conf, u.weightPwr);
			if (e.second.conf > maxConf) {
				maxConf = e.second.conf;
			}
		}

		for (auto &e : transMap) {
			e.second.conf /= maxConf;
			printf("(%d %d)->(%d %d) %e\n", e.first[0][0], e.first[0][1], e.first[1][0], e.first[1][1], e.second.conf);
		}

		writeWeights(u.outDir + "/weights.txt", transMap);

		//rmOutliers(transMap);

		LSQRMatrix A;
		LSQRVectors b;
		makeA(transMap, fetch.szInIms, A);
		makeb(transMap, fetch.szInIms, b);

		storeNewVecs(transMap, fetch.szInIms, A, b, newTrans);
	}

	// the actual positions of all images in imPaths
	vector<float> xs;
	vector<float> ys;
	newTrans.toVecs(xs, ys);
	if (u.usePoslist == false)//don't insert grounding point if we already know the position list
	{
		xs.insert(xs.begin(), 0);
		ys.insert(ys.begin(), 0);
	}

	writePoslist(u.outDir + "/stitch_poslist", xs, ys, fetch);

	string infoFile(u.outDir + "/info.txt");
	ofstream f(infoFile);
	if (!f.is_open()) {
		printf("err: can't open %s for writing\n", infoFile.c_str());
		exit(1);
	}
	f << "total size: " << MAXVEC(xs) - MINVEC(xs) + u.imSz.width
		<< " " << MAXVEC(ys) - MINVEC(ys) + u.imSz.height << endl;
	f.close();
	printf("wrote info at: %s\n", infoFile.c_str());

	// linear blending & tile generation
	//Size tileSz(4096, 4096); // size of output tile in pixels
	TimeSlice tt("Rasterbation:");
	TileHolder tiles(imPaths, xs, ys, u.imSz, u.tileSz);
	Mat tile;
	for (int j = 0; j < tiles.szInTiles.width; j++) {
		for (int i = 0; i < tiles.szInTiles.height; i++) {
			tiles.makeTile(makePt(j, i), fetch, tile, bgIm);

			ostringstream ss;
			ss << u.outDir << "/" << i << "_" << j << "_0";
			string s = ss.str();

			saveFloatIm(s + ".jpg", tile, -0.5, 1.5);
			saveFloatTiff(s + ".tiff", tile);
		}
	}

	return 0;
}

