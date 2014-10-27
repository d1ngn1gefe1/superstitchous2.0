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


#ifndef OPERATOR_TRAIT_HPP_
#define OPERATOR_TRAIT_HPP_

#include "Vector_trait.hpp"

//! @class Operator_trait
/*!
* @brief This class defines the interface of a linear operator to be used in MINRES
*/

class Operator_trait
{
public:
	//! Y = A * X
	void Apply(const Vector_trait & X, Vector_trait & Y) const = 0;
};

//! @class Preconditioner_trait
/*!
 * @brief This class defines the interface of a linear operator to be used as a preconditioner in MINRES
 */
class Preconditioner_trait
{
public:
	//! Y = M \ X;
	void Apply(const Vector_trait & X, Vector_trait & Y) const = 0;
};


#endif /* OPERATOR_TRAIT_HPP_ */
