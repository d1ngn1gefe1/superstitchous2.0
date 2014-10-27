#pragma once
#include <vector>
#include <set>
#include "common.hpp"

using namespace std;

struct LSQRVector {
	vector<float> v;

	LSQRVector(size_t sz);
	inline size_t size() const {
		return v.size();
	}
	inline float &operator[](size_t i) {
		return v[i];
	}
	inline float operator[](size_t i) const {
		return v[i];
	}

	// tminres interface
	LSQRVector &operator=(const float &val);
	void Scale(const float &val);
	LSQRVector *Clone();
};

//! result = v1 + c2*v2
void add(const LSQRVector &v1, const float &c2, const LSQRVector &v2, LSQRVector &result);
//! result = c1*v1 + c2*v2
void add(const float &c1, const LSQRVector &v1, const float &c2, const LSQRVector &v2, LSQRVector &result);
//! result = alpha(v1 + v2)
void add(const float &alpha, const LSQRVector &v1, const LSQRVector &v2, LSQRVector &result);
//! result = v1 + v2 + v3
void add(const LSQRVector &v1, const LSQRVector &v2, const LSQRVector &v3, LSQRVector &result);
//! result = v1 - v2
void subtract(const LSQRVector &v1, const LSQRVector &v2, LSQRVector &result);
//! return the inner product of v1 and v2
float InnerProduct(const LSQRVector &v1, const LSQRVector &v2);

struct LSQROperator {
	int w, h;
	int vecDim; // size of vector to operate on; equals w * h - 1
	PairToTransData transMap;

	LSQROperator(int w, int h, const PairToTransData &transMap);

	// A * x => y
	void Apply(const LSQRVector &x, LSQRVector &y) const;

	void printMat() const;
};

int ptToI(int i, int j, int w);	// convert i, j to index in LSQRVector; 0 <= i < h, 0 <= j < w
