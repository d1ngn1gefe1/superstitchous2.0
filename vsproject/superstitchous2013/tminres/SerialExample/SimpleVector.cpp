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

#include "SimpleVector.hpp"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

SimpleVector::SimpleVector(int size_):
	size(size_)
{
	assert(size > 0);
	vals = new double[size];
}

SimpleVector::~SimpleVector()
{
	delete[] vals;
}

SimpleVector & SimpleVector::operator=(const double & val)
{
	std::fill(vals, vals+size, val);
	return *this;
}
// Set the entry of the Vector equal to the entries in RHS
SimpleVector & SimpleVector::operator=(const SimpleVector & RHS)
{
	std::copy(RHS.vals, RHS.vals + RHS.size, vals);
	return *this;
}

// Set the entries of the vector equal to the vector starting at rhs*
SimpleVector & SimpleVector::operator=(const std::vector<double> RHS)
{
    std::copy(RHS.begin(), RHS.end(), vals);
    return *this;
}


// multiply THIS by a scalar value
void SimpleVector::Scale(const double & val)
{
	for( double * it(vals); it != vals + size; ++it)
		(*it) *= val;
}
// Create a new vector with the same structure of THIS. Values are not initialized.
SimpleVector * SimpleVector::Clone()
{
	return new SimpleVector(size);
}

double & SimpleVector::operator[](const int i)
{
	assert( i < size);
	return vals[i];
}
const double & SimpleVector::operator[](const int i) const
{
	assert( i < size );
	return vals[i];
}

const double SimpleVector::at(const int i) const
{

	if (i<0 || i > size-1)
		return 0.0;

	return vals[i];
}

void SimpleVector::Randomize(int seed)
{
	srand(seed);
	for( double * it(vals); it != vals + size; ++it)
		(*it) = 2.*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1.;

	double norm2( InnerProduct(*this, *this) );
	Scale(1./sqrt(norm2));

}

void SimpleVector::Print(std::ostream & os)
{
	for(double *it(vals); it != vals+size; ++it)
		os << *it << "\t ";

	os << "\n";
}

// result = v1 + c2*v2
void add(const SimpleVector & v1, const double & c2, const SimpleVector & v2, SimpleVector & result)
{
	int size(result.size);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.size == v2.size );
	assert( size == v1.size );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = (*vv1) + c2*(*vv2);
}
// result = c1*v1 + c2*v2
void add(const double & c1, const SimpleVector & v1, const double & c2, const SimpleVector & v2, SimpleVector & result)
{
	int size(result.size);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.size == v2.size );
	assert( size == v1.size );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = c1*(*vv1) + c2*(*vv2);
}

// result = alpha(v1 + v2)
void add(const double & alpha, const SimpleVector & v1, const SimpleVector & v2, SimpleVector & result)
{
	int size(result.size);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.size == v2.size );
	assert( size == v1.size );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = alpha*(*vv1 +*vv2);

}

// result = v1 + v2 + v3
void add(const SimpleVector & v1, const SimpleVector & v2, const SimpleVector & v3, SimpleVector & result)
{
	int size(result.size);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);
	double * vv3(v3.vals);

	assert( v1.size == v2.size );
	assert( v2.size == v3.size );
	assert( size == v1.size );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2, ++vv3)
		(*rr) = (*vv1) + (*vv2) + (*vv3);

}
// result = v1 - v2
void subtract(const SimpleVector & v1, const SimpleVector & v2, SimpleVector & result)
{
	int size(result.size);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.size == v2.size );
	assert( size == v1.size );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = *vv1  - *vv2;


}

// return the inner product of v1 and v2
double InnerProduct(const SimpleVector & v1, const SimpleVector & v2)
{
	double result(0);
	int size(v1.size);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.size == v2.size );

	double * end(vv1 + size);

	for( ; vv1 != end; ++vv1, ++vv2)
		result += (*vv1) * (*vv2);

	return result;

}

