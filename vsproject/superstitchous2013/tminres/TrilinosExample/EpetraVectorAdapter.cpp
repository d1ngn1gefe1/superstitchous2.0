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

#include "EpetraVectorAdapter.hpp"

EpetraVectorAdapter::EpetraVectorAdapter(Epetra_MultiVector & v_):
	v(&v_),
	ownEpetraVector(false)
{
	assert( v->NumVectors() == 1);
	v->ExtractView(&vals, &localSize);
}

EpetraVectorAdapter::EpetraVectorAdapter(Epetra_MultiVector * v_, bool owned):
	v(v_),
	ownEpetraVector(owned)
{
	assert( v->NumVectors() == 1);
	v->ExtractView(&vals, &localSize);
}

EpetraVectorAdapter::~EpetraVectorAdapter()
{
	if(ownEpetraVector)
		delete v;
}

//! Set all the entry of the Vector equal to val
EpetraVectorAdapter & EpetraVectorAdapter::operator=(const double & val)
{
	std::fill(vals, vals+localSize, val);
	return *this;
}

//! Set the entry of the Vector equal to the entries in RHS
EpetraVectorAdapter & EpetraVectorAdapter::operator=(const EpetraVectorAdapter & RHS)
{
	assert( localSize == RHS.localSize);
	std::copy(RHS.vals, RHS.vals+ RHS.localSize, vals);
	return *this;
}

//! multiply THIS by a scalar value
void EpetraVectorAdapter::Scale(const double & val)
{
	for( double * it(vals); it != vals + localSize; ++it)
		(*it) *= val;
}

void EpetraVectorAdapter::Randomize(int seed)
{
	v->SetSeed( seed );
	v->Random();
	double norm2( InnerProduct(*this, *this) );
	Scale(1./sqrt(norm2));

}


//! Create a new vector with the same structure of THIS. Values are not initialized.
EpetraVectorAdapter * EpetraVectorAdapter::Clone()
{
	Epetra_MultiVector * cv(new Epetra_MultiVector(v->Map(), v->NumVectors()) );
	return new EpetraVectorAdapter(cv);
}

Epetra_MultiVector & EpetraVectorAdapter::EpetraVector()
{
	return *v;
}

const Epetra_MultiVector & EpetraVectorAdapter::EpetraVector() const
{
	return *v;
}

void EpetraVectorAdapter::Print(std::ostream & os) const
{
	double * end(vals+localSize);
	for(double * it(vals); it != end; ++it)
		os << *it << "\t";
	os << "\n";
}

//! result = v1 + c2*v2
void add(const EpetraVectorAdapter & v1, const double & c2, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result)
{
	int size(result.localSize);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.localSize == v2.localSize );
	assert( size == v1.localSize );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = (*vv1) + c2*(*vv2);

}
//! result = c1*v1 + c2*v2
void add(const double & c1, const EpetraVectorAdapter & v1, const double & c2, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result)
{
	int size(result.localSize);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.localSize == v2.localSize );
	assert( size == v1.localSize );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = c1*(*vv1) + c2*(*vv2);


}
//! result = alpha(v1 + v2)
void add(const double & alpha, const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result)
{
	int size(result.localSize);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.localSize == v2.localSize );
	assert( size == v1.localSize );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = alpha*(*vv1 + *vv2);

}
//! result = v1 + v2 + v3
void add(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, const EpetraVectorAdapter & v3, EpetraVectorAdapter & result)
{
	int size(result.localSize);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);
	double * vv3(v3.vals);

	assert( v1.localSize == v2.localSize );
	assert( v1.localSize == v3.localSize );
	assert( size == v1.localSize );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2, ++vv3)
		(*rr) = (*vv1) + (*vv2) + (*vv3);


}
//! result = v1 - v2
void subtract(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2, EpetraVectorAdapter & result)
{
	int size(result.localSize);
	double * rr(result.vals);
	double * vv1(v1.vals);
	double * vv2(v2.vals);

	assert( v1.localSize == v2.localSize );
	assert( size == v1.localSize );

	double * end(rr + size);

	for( ; rr != end; ++rr, ++vv1, ++vv2)
		(*rr) = (*vv1 - *vv2);

}
//! return the inner product of v1 and v2
double InnerProduct(const EpetraVectorAdapter & v1, const EpetraVectorAdapter & v2)
{
	assert(v1.v->NumVectors() == 1);
	assert(v2.v->NumVectors() == 1);

	double result;
	v1.v->Dot(*v2.v, &result);

	return result;
}
