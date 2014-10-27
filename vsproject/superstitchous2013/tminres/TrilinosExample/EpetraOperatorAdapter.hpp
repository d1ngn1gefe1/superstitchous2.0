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

#ifndef EPETRAOPERATORADAPTER_HPP_
#define EPETRAOPERATORADAPTER_HPP_

#include "EpetraVectorAdapter.hpp"
#include <Epetra_Operator.h>

//! @class EpetraOperatorAdapter
/*!
 * @brief Wrap an Epetra_Operator in a operator for MINRES
 */
class EpetraOperatorAdapter
{
public:
	//! Construnctor
	/*!
	 * @param op_ Epetra_Operator
	 */
	EpetraOperatorAdapter(Epetra_Operator & op_) : op(op_) { };

	//! Y = op * X
	void Apply(const EpetraVectorAdapter & X, EpetraVectorAdapter & Y) const
	{
		int ierr(0);
		ierr = op.Apply(X.EpetraVector(), Y.EpetraVector() );

		assert(0 == ierr);
	}

	virtual ~EpetraOperatorAdapter(){ };

private:
	Epetra_Operator & op;
};

//! @class EpetraPreconditionerAdapter
/*!
 * @brief Wrap an Epetra_Operator in a preconditioner for MINRES
 */
class EpetraPreconditionerAdapter
{
public:

	//! Construnctor
	/*!
	 * @param prec_ Epetra_Operator
	 */
	EpetraPreconditionerAdapter(Epetra_Operator & prec_) : prec(prec_) { };

	//! Y = prec \ X
	void Apply(const EpetraVectorAdapter & X, EpetraVectorAdapter & Y) const
	{
		int ierr(0);
		ierr = prec.ApplyInverse(X.EpetraVector(), Y.EpetraVector() );

		assert( 0 == ierr );
	}

	virtual ~EpetraPreconditionerAdapter(){ };

private:
	Epetra_Operator & prec;
};


#endif /* EPETRAOPERATORADAPTER_HPP_ */
