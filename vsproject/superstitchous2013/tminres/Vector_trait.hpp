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

#ifndef VECTOR_TRAIT_HPP_
#define VECTOR_TRAIT_HPP_

//! @class Vector_trait
/*!
* @brief This class describes the interface of a VECTOR to be used in minres.
*/

class Vector_trait
{

	//! Set all the entry of the Vector equal to val
	Vector_trait & operator=(const double & val) = 0;
	//! Set the entry of the Vector equal to the entries in RHS
	Vector_trait & operator=(const Vector_trait & RHS) = 0;
	//! multiply THIS by a scalar value
	void Scale(const double & val) = 0;
	//! Create a new vector with the same structure of THIS. Values are not initialized.
	Vector_trait * Clone() = 0;

	//! result = v1 + c2*v2
	friend void add(const Vector_trait & v1, const double & c2, const Vector_trait & v2, Vector_trait & result);
	//! result = c1*v1 + c2*v2
	friend void add(const double & c1, const Vector_trait & v1, const double & c2, const Vector_trait & v2, Vector_trait & result);
	//! result = alpha(v1 + v2)
	friend void add(const double & alpha, const Vector_trait & v1, const Vector_trait & v2, Vector_trait & result);
	//! result = v1 + v2 + v3
	friend void add(const Vector_trait & v1, const Vector_trait & v2, const Vector_trait & v3, Vector_trait & result);
	//! result = v1 - v2
	friend void subtract(const Vector_trait & v1, const Vector_trait & v2, Vector_trait & result);
	//! return the inner product of v1 and v2
	friend double InnerProduct(const Vector_trait & v1, const Vector_trait & v2);



};


#endif /* VECTOR_TRAIT_HPP_ */
