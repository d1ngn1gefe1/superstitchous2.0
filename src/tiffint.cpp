#define _CRT_SECURE_NO_WARNINGS
#include "tiffint.hpp"
#include "tiffio.h"
#include <stdio.h>
#include <stdlib.h>

void saveFloatTiff(const string &name, const float *buf, int w, int h) {
	TIFF *out = TIFFOpen(name.c_str(), "w");
	if (!out) {
		TIFFError("tiffint", NULL);
	}

	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, w);
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, h);
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, h);

	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);
	TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
	TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

	TIFFSetField(out, TIFFTAG_SOFTWARE, "ECE_445");
	TIFFSetField(out, TIFFTAG_MAKE, "Kevin&Zelun");
	TIFFSetField(out, TIFFTAG_MODEL, __DATE__);

	//	printf("tile size: %ld, n tiles: %d, check: %d\n", TIFFTileSize(out), TIFFNumberOfTiles(out), TIFFCheckTile(out, 0, 0, 0, 0));
	printf("saving %s\n", name.c_str());
	if (TIFFWriteEncodedStrip(out, 0, (void *)buf, w * h * sizeof(float)) == -1) {
		TIFFError("tiffint", NULL);
		exit(1);
	}

	TIFFClose(out);
}

void loadFloatTiff(const string &name, float *buf)
{
	TIFF* in = TIFFOpen(name.c_str(), "r");
	if (!in) {
		TIFFError("tiffint", NULL);
	}

	int w, h, a;
	TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(in, TIFFTAG_IMAGELENGTH, &h);
	TIFFGetField(in, TIFFTAG_ROWSPERSTRIP, &a);
	tsize_t scanline = TIFFScanlineSize(in);

	//printf("row per strip: %d\n", a);

	for (int row = 0; row < h; row++) {
		TIFFReadScanline(in, buf+scanline/sizeof(float)*row, row);
	}

	TIFFClose(in);
}

void loadFloatTiffFast(const string &name, float *buf, int w, int h)
{
	FILE *in = fopen(name.c_str(), "rb");
	if (!in) {
		printf("err: can't read %s\n", name.c_str());
	}
	fseek(in, 8, SEEK_SET);
	auto checkme = fread(buf, sizeof(float), w * h, in);
	if (checkme!=w*h)
	{
		printf("Warning: %s is corrupt",name.c_str());
	}
	fclose(in);
}
