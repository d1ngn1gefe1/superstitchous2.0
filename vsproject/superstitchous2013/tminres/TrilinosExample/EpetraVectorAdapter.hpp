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

#ifndef EPETRAVECTORADAPTER_HPP_
#define EPETRAVECTORADAPTER_HPP_

#include <Epetra_MultiVector.h>
#include <iostream>

//! @class EpetraVectorAdapter
/*!
 * @brief Wrapper class to use Trilinos Epetra_Vector with minres.
 */

class EpetraVectorAdapter
{
public:
	//! Wrap the Epetra_MultiVector v_.
	/*!
	 * If the object v_ is distroyed, "this" will have invalid pointers.
	 */
	EpetraVectorAdapter(Epetra_MultiVector & v_);

	//! Wrap the Epetra_MultiVector v_
	/*!
	 * If owned == true, this class will deallocate the object v_ in the Destructor.
	 */
	EpetraVectorAdapter(Epetra_MultiVector * v_, bool owned = true);

	//! Destructor
	/*
	 * if owned == true, dellacate the memory pointed by v.
	 */
	virtual ~EpetraVectorAdapter();

	//! Set all the entry of the Vector equal to val
	EpetraVectorAdapter & operator=(const double & val);
	//! Set the entry of the Vector equal to the entries in RHS
	EpetraVectorAdapter & operator=(const EpetraVectorAdapter & RHS);
	//! multiply THIS by a scalar value
	void Scale(const double & val);
	//! Fill with Random entries (|| this ||_2 = 1)
	void Randomize(int seed);
	//! Create a new vector with the same structure of THIS. Values are not initialized.
	EpetraVectorAdapter * Clone();

	//! Extract the Epetra_Vector object v
	Epetra_MultiVector & EpetraVector();
	//! Extract the Epetra_Vector object v (constant version)
	const Epetra_MultiVector & EpetraVector() const;

	//! Print my Local entries
	void Print(std::ostream & os) const;

	//! result = v1 + c2*v2
	friend void add(const EpetraVectorAdapter & v1, const double & c2, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result);
	//! result = c1*v1 + c2*v2
	friend void add(const double & c1, const EpetraVectorAdapter & v1, const double & c2, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result);
	//! result = alpha(v1 + v2)
	friend void add(const double & alpha, const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result);
	//! result = v1 + v2 + v3
	friend void add(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, const EpetraVectorAdapter & v3, EpetraVectorAdapter & result);
	//! result = v1 - v2
	friend void subtract(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result);
	//! return the inner product of v1 and v2
	friend double InnerProduct(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2);

private:
	Epetra_MultiVector * v;
	bool ownEpetraVector;
	double * vals;
	int localSize;


};

#endif /* EPETRAVECTORADAPTER_HPP_ */
