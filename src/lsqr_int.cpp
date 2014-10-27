#include "lsqr_int.hpp"
#include <cassert>

LSQRVector &LSQRVector::operator=(const float &val) {
	for (size_t i = 0; i < v.size(); i++) {
		v[i] = val;
	}
	return *this;
}

void LSQRVector::Scale(const float &val) {
	for (size_t i = 0; i < v.size(); i++) {
		v[i] *= val;
	}
}

LSQRVector *LSQRVector::Clone() {
	LSQRVector *newV = new LSQRVector(v.size());
	return newV;
}

LSQRVector::LSQRVector(size_t sz) : v(sz) {
}

void add(const LSQRVector &v1, const float &c2, const LSQRVector &v2, LSQRVector &result) {
	for (size_t i = 0; i < v1.size(); i++) {
		result[i] = v1[i] + c2 * v2[i];
	}
}

void add(const float &c1, const LSQRVector &v1, const float &c2, const LSQRVector &v2, LSQRVector &result) {
	for (size_t i = 0; i < v1.size(); i++) {
		result[i] = c1 * v1[i] + c2 * v2[i];
	}
}

void add(const float &alpha, const LSQRVector &v1, const LSQRVector &v2, LSQRVector &result) {
	for (size_t i = 0; i < v1.size(); i++) {
		result[i] = alpha * (v1[i] + v2[i]);
	}
}

void add(const LSQRVector &v1, const LSQRVector &v2, const LSQRVector &v3, LSQRVector &result) {
	for (size_t i = 0; i < v1.size(); i++) {
		result[i] = v1[i] + v2[i] + v3[i];
	}
}

void subtract(const LSQRVector &v1, const LSQRVector &v2, LSQRVector &result) {
	for (size_t i = 0; i < v1.size(); i++) {
		result[i] = v1[i] - v2[i];
	}
}

float InnerProduct(const LSQRVector &v1, const LSQRVector &v2) {
	float s = 0;
	for (size_t i = 0; i < v1.size(); i++) {
		s += v1[i] * v2[i];
	}
	return s;
}

LSQROperator::LSQROperator(int w, int h, const PairToTransData &transMap) :
		w(w),
		h(h),
		vecDim(w * h - 1),
		transMap(transMap) {
}

int ptToI(int i, int j, int w) {
	return i * w + j - 1;
}

bool lookup(const set<PtPair> &missingPairs, const PtPair &pair) {
	PtPair swapped = {{pair[1], pair[0]}};
	return missingPairs.count(pair) || missingPairs.count(swapped);
}

// A * x => y
void LSQROperator::Apply(const LSQRVector &x, LSQRVector &y) const {
	assert(x.size() == (unsigned)vecDim && y.size() == (unsigned)vecDim);
	unsigned outI = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (j == 0 && i == 0) {
				continue;
			}

			float s = 0;

			unsigned nbrs = 0;

			for (int i1 = i - 1; i1 < i + 2; i1++) {
				for (int j1 = j - 1; j1 < j + 2; j1++) {
					if (i1 < 0 || i1 == h || j1 < 0 || j1 == w || (j1 == j && i1 == i)) {
						continue;
					}
					PtPair pair = {{makePt(j, i), makePt(j1, i1)}};
					TransData dat;
					if (!lookupPair(transMap, pair, dat)) {
						printf("err: pair not found: (%d, %d), (%d, %d)\n", j, i, j1, i1);
						exit(1);
					}

					if (j1 != 0 || i1 != 0) {	// since ptToI(0, 0, w) doesn't exist
						s += (x[ptToI(i, j, w)] - x[ptToI(i1, j1, w)]) * dat.conf;
					} else {
						s += x[ptToI(i, j, w)] * dat.conf;
					}

					if (dat.conf) //dangerous? ~MK
					{
						nbrs++;
					}
				}
			}

			if (nbrs == 0) {
//				printf("err: (%u, %u) has no neighbors\n", j, i);
//				exit(1);
				printf("warning: (%d, %d) has no neighbors\n", j, i);
			}

			y[outI++] = s;
		}
	}
}

void LSQROperator::printMat() const {
	LSQRVector v(vecDim);
	printf("Mat (transposed):\n");
	for (int i = 0; i < vecDim; i++) {
		v[i] = 1;

		LSQRVector out(vecDim);
		Apply(v, out);

		for (int j = 0; j < out.size(); j++) {
			printf("%f ", out[j]);
		}
		printf("\n");

		v[i] = 0;
	}
}

//#include <cstdio>
//int main(int argc, const char *argv[]) {
//	int n = 4;
//	int vecDim = n * n - 1;
//	LSQRVector a(vecDim);
//	for (int i = 0; i < vecDim; i++) {
//		a[i] = 1;
//
//		LSQRVector out(vecDim);
//		LSQROperator o(n);
//		o.Apply(a, out);
//		for (int j = 0; j < vecDim; j++) {
//			printf("%f ", out[j]);
//		}
//		printf("\n");
//		a[i] = 0;
//	}
//	return 0;
//}
