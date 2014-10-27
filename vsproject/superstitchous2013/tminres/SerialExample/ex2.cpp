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

/*!
 * @example SerialExample/ex2.cpp
 *
 * A simple example of MINRES() usage with diagonal preconditioner.
 *
 * We consider a Markers and Cells Finite Volume discretization of a 2d Stokes problem:
 * @f{eqnarray*}{
 * \displaystyle \mathbf{u} - \Delta \mathbf{u} + \nabla p &= \mathbf{f} & {\rm in} \; [0,1]\times[0,1] \\
 * \displaystyle {\rm div}\; \mathbf{u} &= 0 & {\rm in} \; [0,1]\times[0,1]
 * @f}
 *
 * where @f$\mathbf{u} = [u,v]@f$ is the velocity field and @f$p@f$ the pressure field.
 *
 * Component-wise the Stokes Equations read:
 * \f{eqnarray*}{
 * \displaystyle u - \Delta u + \frac{dp}{dx} &=& f_x \\
 * \displaystyle v - \Delta v + \frac{dp}{dy} &=& f_y \\
 * \displaystyle \frac{du}{dx} + \frac{dv}{dy}&=& 0
 * \f}
 *
 */
#include <tminres.hpp>
#include "SimpleVector.hpp"
#include <cmath>
#include <cassert>

//! @class StokesOperator
/*!
 * @brief A matrix free operator for the generalized Stokes problem
 *
*/


class StokesOperator {
public:

	//! Constructor
	/*!
	 * @param n_ int : number of cell in the x direction. The number of cell in the y direction is also n_.
	 */
	StokesOperator(int n_ );

	//! Do nothing Destructor.
	virtual ~StokesOperator();

	//! Y = A*X where A is the Finite Volumes matrix of the discrete system.
	void Apply(const SimpleVector & X, SimpleVector & Y) const;

private:

	inline int getUIndex(int i, int j) const
	{
		if(i == -1)
			i = n;

		if(i==n+1)
			i = 0;

		if(j== -1)
			j = n-1;

		if(j==n)
			j = 0;

		return offsetU + i + (n+1)*j;
	}

	inline int getVIndex(int i, int j) const
	{
		if(i == -1)
			i = n-1;

		if(i==n)
			i = 0;

		if(j== -1)
			j = n;

		if(j==n+1)
			j = 0;

		return offsetV + i + n*j;
	}

	inline int getPIndex(int i, int j) const
	{
		if(i == -1 || i == n || j == -1 || j == n)
			return -1;

		return offsetP + i + n*j;
	}


	int n;
	double h;
	int offsetU;
	int offsetV;
	int offsetP;
};

//! @class StokesDiagonalScaling
/*!
 * @brief A simple Diagonal preconditioner for the generalized Stokes Problem
 */
class StokesDiagonalScaling
{
public:
	//! Constructor
	/*!
	 * @param n_ int : number of cell in the x direction. The number of cell in the y direction is also n_.
	 */
	StokesDiagonalScaling(int n_);

	virtual ~StokesDiagonalScaling();

	//! Y = M\X where M is the diagonal of the matrix in StokesOperator.
	void Apply(const SimpleVector & X, SimpleVector & Y) const;

private:
	int n;
	double h;
	int offsetU;
	int offsetV;
	int offsetP;

};



int main()
{
	//(1) Discretization parameters and problem size
	int n(4);					//number of cell for each edge
	int dim(n*n + 2*(n+1)*n);	//total number of unknows (n^2 pressure and (n+1)n for each velocity component)
	//(2) Define the linear operator "op" we want to solve.
	StokesOperator op(n);
	//(3) Define the exact solution (at random)
	SimpleVector sol(dim);
	sol.Randomize( 1 );

	//(4) Define the "rhs" as "rhs = op*sol"
	SimpleVector rhs(dim);
	op.Apply(sol, rhs);
	double rhsNorm( sqrt(InnerProduct(rhs,rhs)) );
	std::cout << "|| rhs || = " << rhsNorm << "\n";

	//(5) Define the preconditioner
	StokesDiagonalScaling prec(n);

	//(6) Use an identically zero initial guess
	SimpleVector x(dim);
	x = 0;

	//(7) Set the minres parameters
	double shift(0);
	int max_iter(100);
	double tol(1e-6);
	bool show(true);

	//(8) Solve the problem with minres
	MINRES(op, x, rhs, &prec, shift, max_iter, tol, show);

	//(9) Compute the error || x_ex - x_minres ||_2
	subtract(x, sol, x);
	double err2 = InnerProduct(x,x);

	std::cout<< "|| x_ex - x_n || = " << sqrt(err2) << "\n";

	return 0;
}


StokesOperator::StokesOperator(int n_):
	n(n_),
	h(1./static_cast<double>(n_)),
	offsetU(0),
	offsetV( (n+1)*n ),
	offsetP( (n+1)*n + n*(n+1) )
{
	assert(n > 0);
}

StokesOperator::~StokesOperator()
{
	// TODO Auto-generated destructor stub
}

void StokesOperator::Apply(const SimpleVector & X, SimpleVector & Y) const
{
	Y = 0.;
	for(int i(0); i<n+1; ++i)
		for(int j(0); j<n+1; ++j)
		{
			double uij ( X[getUIndex(i  ,j  ) ] );
			double uimj( X[getUIndex(i-1,j  ) ] );
			double uipj( X[getUIndex(i+1,j  ) ] );
			double uijm( X[getUIndex(i  ,j-1) ] );
			double uijp( X[getUIndex(i  ,j+1) ] );

			double vij ( X[getVIndex(i  ,j  ) ] );
			double vimj( X[getVIndex(i-1,j  ) ] );
			double vipj( X[getVIndex(i+1,j  ) ] );
			double vijm( X[getVIndex(i  ,j-1) ] );
			double vijp( X[getVIndex(i  ,j+1) ] );

			double pij ( X.at(getPIndex(i  ,j  ) ) );
			double pimj( X.at(getPIndex(i-1,j  ) ) );
			double pijm( X.at(getPIndex(i  ,j-1) ) );

			double laplacianu( 4.*uij - uimj - uipj - uijm - uijp );
			laplacianu *= (1/h/h);
			double laplacianv( 4.*vij - vimj - vipj - vijm - vijp );
			laplacianv *= (1/h/h);

			double dxp( pij - pimj);
			dxp *= 1./h;
			double dyp( pijm - pij);
			dyp *= 1./h;

			double div( uipj - uij + vij - vijp);
			div *= -1./h;

			if ( j != n )
				Y[getUIndex(i,j)] += laplacianu + dxp + uij;

			if ( i !=n )
				Y[getVIndex(i,j)] += laplacianv + dyp + vij;

			if( i != n && j != n )
				Y[getPIndex(i,j)] += div;


		}
}


StokesDiagonalScaling::StokesDiagonalScaling(int n_):
	n(n_),
	h(1./static_cast<double>(n_)),
	offsetU(0),
	offsetV( (n+1)*n ),
	offsetP( (n+1)*n + n*(n+1) )
{
	assert(n > 0);
}


StokesDiagonalScaling::~StokesDiagonalScaling()
{

}

void StokesDiagonalScaling::Apply(const SimpleVector & X, SimpleVector & Y) const
{
	int i(0);

	double diagValue(4./h/h + 1.);
	for( ; i < offsetP; ++i)
		Y[i] = X[i]/diagValue;

	for( ; i < offsetP + n*n; ++i)
		Y[i] = X[i];
}
