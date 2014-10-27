#pragma once
#include <string>

using namespace std;

// save tiff in 32-bit float format with libtiff
void saveFloatTiff(const string &name, const float *buf, int w, int h);

// load 32-bit tif file with libtiff
void loadFloatTiff(const string &name, float *buf);

// faster than loadFloatTiff if the image is continuous
void loadFloatTiffFast(const string &name, float *buf, int w, int h);