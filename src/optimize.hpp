#pragma once
#include <map>
#include <array>
#include "Eigen/Sparse"
#include "fetcher.hpp"
#include "common.hpp"

using namespace std;
using namespace Eigen;

typedef SparseMatrix<float> LSQRMatrix;
typedef BiCGSTAB<LSQRMatrix, IncompleteLUT<float>> LSQRSolver;

// least square
struct LSQRVectors {
	VectorXf xs;
	VectorXf ys;

	LSQRVectors();
	LSQRVectors(Size szInIms);
	LSQRVectors(int numIms);

	void toVecs(vector<float> &xs, vector<float> &ys);

	inline size_t size() const {
		return xs.size();
	}

	void push_back(float x, float y);
	void resize(size_t sz);
	void clear();

	void print();
};

void makeA(const PairToTransData &transMap,const Size& szInIms, LSQRMatrix &A);
void makeb(const PairToTransData &transMap,const Size& szInIms, LSQRVectors &b);

void storeNewVecs(const PairToTransData &transMap,const Size& szInIms, const LSQRMatrix &A, const LSQRVectors &b, LSQRVectors &x);
