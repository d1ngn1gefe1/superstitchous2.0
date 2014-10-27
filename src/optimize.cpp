#include <cstdio>
#include <cstdlib>
#include "optimize.hpp"
#include "tminres/tminres.hpp"

//static bool lookup(const PairToTransData &transMap, const GridPt &p1, const GridPt &p2, Point &out) {
//	auto it = transMap.find(makePair(p1, p2));
//	if (it == transMap.end()) {
//		it = transMap.find(makePair(p2, p1));
//		if (it == transMap.end()) {
//			return false;
//		}
//		out = Point(-it->second.trans.x, -it->second.trans.y);
//	} else {
//		out = it->second.trans;
//	}
//	return true;
//}

static int vecLen(const Size& szInIms) {
	return szInIms.area() - 1;
}

static int ptToI(int i, int j, int w) {
	return i * w + j - 1;
}

static void chkSolver(const LSQRSolver &solver) {
	auto info = solver.info();
	if (info != Success) {
		cout << "Solver failed with info: " << info << endl;
		exit(1);
	}
}

LSQRVectors::LSQRVectors() {
}

LSQRVectors::LSQRVectors(Size szInIms) :
		xs(vecLen(szInIms)),
		ys(vecLen(szInIms)) {
}

LSQRVectors::LSQRVectors(int numIms) :
		xs(numIms - 1),
		ys(numIms - 1) {
}

void LSQRVectors::toVecs(vector<float> &xs, vector<float> &ys) {
	xs.resize(this->xs.size());
	ys.resize(this->ys.size());
	for (int i = 0; i < this->xs.size(); i++) {
		xs[i] = this->xs[i];
		ys[i] = this->ys[i];
	}
}

void LSQRVectors::push_back(float x, float y) {
	xs.conservativeResize(xs.rows() + 1, 1);
	ys.conservativeResize(ys.rows() + 1, 1);
	xs[xs.rows() - 1] = x;
	ys[ys.rows() - 1] = y;
}

void LSQRVectors::resize(size_t sz) {
	xs.resize(sz);
	ys.resize(sz);
}

void LSQRVectors::clear() {
	resize(0);
}

void LSQRVectors::print() {
	printf("xs: ");
	for (int i = 0; i < xs.size(); i++) {
		printf("%f ", xs[i]);
	}
	printf("\nys: ");
	for (int i = 0; i < ys.size(); i++) {
		printf("%f ", ys[i]);
	}
	printf("\n");
}

void makeA(const PairToTransData &transMap,const Size& szInIms, LSQRMatrix &A) {
	typedef Triplet<float> T;

	vector<T> trips;
	int row = 0;
	for (int i = 0; i < szInIms.height; i++) {
		for (int j = 0; j < szInIms.width; j++) {
			if (i == 0 && j == 0) {
				continue;
			}

			float s = 0;
			unsigned nbrs = 0;
			for (int i1 = i - 1; i1 <= i + 1; i1++) {
				for (int j1 = j - 1; j1 <= j + 1; j1++) {
					if (j1 < 0 || j1 >= szInIms.width || i1 < 0 || i1 >= szInIms.height || (j == j1 && i == i1)) {
						continue;
					}

					PtPair pair = {{makePt(j, i), makePt(j1, i1)}};
					TransData dat;
					if (!lookupPair(transMap, pair, dat)) {
						printf("err: pair not found: (%d, %d), (%d, %d)\n", j, i, j1, i1);
						exit(1);
					}

					if (j1 != 0 || i1 != 0) {	// since ptToI(0, 0, w) doesn't exist
//						printf("%d %d %f\n", row, ptToI(i1, j1, szInIms.width), -dat.conf);
						trips.push_back(T(row, ptToI(i1, j1, szInIms.width), -dat.conf));
					}
					s += dat.conf;

					if (dat.conf) {
						nbrs++;
					}
				}
			}

			if (nbrs == 0) {
//				printf("err: (%u, %u) has no neighbors\n", j, i);
//				exit(1);
				printf("warning: (%d, %d) has no neighbors\n", j, i);
			}

//			printf("%d %d %f\n", row, ptToI(i, j, szInIms.width), s);
			trips.push_back(T(row, ptToI(i, j, szInIms.width), s));

			row++;
		}
	}

	int vecDim = vecLen(szInIms);
	A.resize(vecDim, vecDim);
	A.setFromTriplets(trips.begin(), trips.end());
}

void makeb(const PairToTransData &transMap,const Size& szInIms, LSQRVectors &b) {
	b.resize(vecLen(szInIms));

	unsigned outI = 0;

	for (int y = 0; y < szInIms.height; y++) {
		for (int x = 0; x < szInIms.width; x++) {
			if (x == 0 && y == 0) {
				continue;
			}

			float xSum = 0, ySum = 0;
			for (int x1 = x - 1; x1 < x + 2; x1++) {
				for (int y1 = y - 1; y1 < y + 2; y1++) {
					if (x1 < 0 || x1 == szInIms.width || y1 < 0 || y1 == szInIms.height || (x1 == x && y1 == y)) {
						continue;
					}
					PtPair pair = {{makePt(x1, y1), makePt(x, y)}};
					TransData dat;
					bool swapped;
					if (lookupPair(transMap, pair, dat, &swapped)) {
						if (swapped) {
							dat.trans.x = -dat.trans.x;
							dat.trans.y = -dat.trans.y;
						}
						xSum += dat.trans.x * dat.conf;
						ySum += dat.trans.y * dat.conf;
					} else {
						printf("missing translation between (%d, %d) and (%d, %d)\n", pair[0][0], pair[0][1], pair[1][0], pair[1][1]);
						exit(1);
					}
				}
			}
			b.xs[outI] = xSum;
			b.ys[outI] = ySum;
			outI++;
		}
	}
}

void storeNewVecs(const PairToTransData &transMap,const Size& szInIms, const LSQRMatrix &A, const LSQRVectors &b, LSQRVectors &x) {
	assert(A.rows() == vecLen(szInIms) && A.cols() == vecLen(szInIms) && b.size() == vecLen(szInIms));

	LSQRSolver solver;
	solver.setMaxIterations(100000);

	printf("factorizing...\n"); 
	solver.compute(A);
	chkSolver(solver);

	cout << b.xs << endl;
	printf("solving xs...\n");
	x.xs = solver.solve(b.xs);
	chkSolver(solver);

	cout << b.ys << endl;
	printf("solving ys...\n");
	x.ys = solver.solve(b.ys);
	chkSolver(solver);
}
