// tminres is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
// Authors:
// - Umberto Villa, Emory University - uvilla@emory.edu
// - Michael Saunders, Stanford University
// - Santiago Akle, Stanford University

/*!
@file
@author U. Villa - uvilla@emory.edu
@date 04/2012
*/

#include <tminres.hpp>
#include "SimpleVector.hpp"
#include <cmath>
#include <fstream>

//!@class Preconditioner
/*
 * @brief An abstract preconditioner class.
 * It should be used to call minres without preconditioner
 */
class Preconditioner
{
public:
	//! Y = M\X
	virtual void Apply(const SimpleVector & X, SimpleVector & Y) const = 0;
};

//!@class Operator
/*
 * @brief A simple diagonal linear operator.
 * The first half of the diagonal entries are positive, the second half are negative
 */
class Operator
{
public:

	//!Constructor
	/*
	 * @param size_ problem size
	 */
	Operator(int size_) :
		size(size_)
	{
		vals = new double[size];
		int i(0);
		for( ; i < size/2; ++i)
			vals[i] = i+1;

		for( ; i < size; ++i)
			vals[i] = -i;
	}

	~Operator()
	{
		delete[] vals;
	}

	//! Y = A*X;
	void Apply(const SimpleVector & X, SimpleVector & Y) const
	{
		for(int i(0); i < size; ++i)
			Y[i] = vals[i] * X[i];
	}

	void Print(std::ostream & os)
	{
		os<< "d = [";
		int i(0);
		for( ; i<size-1; ++i)
			os<<vals[i]<<"; ";
		os<<vals[i]<<"];\n";
		os<<"Op = spdiags(d, 0,"<<size<<", "<<size<<");\n";
	}

private:
	int size;
	double *vals;
};


/*!
 * \example SerialExample/ex1.cpp
 *
 * A simple example of MINRES() usage without preconditioner.
 */
int main()
{
	//(1) Define the size of the problem we want to solve
	int size(10);
	//(2) Define the linear operator "op" we want to solve.
	Operator op(size);
	//(3) Define the exact solution (at random)
	SimpleVector sol(size);
	sol.Randomize( 1 );

	//(4) Define the "rhs" as "rhs = op*sol"
	SimpleVector rhs(size);
	op.Apply(sol, rhs);
	double rhsNorm( sqrt(InnerProduct(rhs,rhs)) );
	std::cout << "|| rhs || = " << rhsNorm << "\n";

	//(5) We don't use any preconditioner. Let prec be a null pointer.
	Preconditioner * prec = NULL;

	//(6) Use an identically zero initial guess
	SimpleVector x(size);
	x = 0;

	//(7) Set the minres parameters
	double shift(0);
	int max_iter(100);
	double tol(1e-6);
	bool show(true);

	//(8) Solve the problem with minres
	MINRES(op, x, rhs, prec, shift, max_iter, tol, show);

	//(9) Compute the error || x_ex - x_minres ||_2
	subtract(x, sol, x);
	double err2 = InnerProduct(x,x);
	std::cout<< "|| x_ex - x_n || = " << sqrt(err2) << "\n";

	std::ofstream fid("ex1.m");
	op.Print(fid);
	fid<< "rhs = [";
	for(int i(0); i<size-1; ++i)
		fid<<rhs[i] <<"; ";
	fid<<rhs[size-1] <<"]; \n";
	fid<<"[ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = minres(Op, rhs, [], 0, true, false, 100, 1e-6);\n";


	return 0;
}
