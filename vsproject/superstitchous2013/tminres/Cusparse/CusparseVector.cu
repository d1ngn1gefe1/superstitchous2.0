/*
 * CusparseVector.cu
 *
 *  Created on: Aug 2012
 *      Author: Santiago Akle
 */

#include "CusparseVector.hpp"
#include <cusparse_v2.h>

#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

CusparseVector::CusparseVector(cusparseHandle_t & handle_, int size_):
	size(size_), handle(handle_)
{
    assert(size > 0);
    //Allocate the space in the GPU
    cudaError_t err = cudaMalloc((void**)&d_v,size*sizeof(double));
    if(err!=cudaSuccess)
    {
       fprintf(stderr,"Error: unable o allocate vector\n");
       throw err;
    }
    //Make a thrust device pointer for the thrust operations
    td_v = thrust::device_pointer_cast(d_v);	
}


CusparseVector::~CusparseVector()
{
	cudaError_t err = cudaFree(d_v);
    if (err != cudaSuccess) {
        fprintf (stderr, "!!!! device access error (free Vector)\n");
        throw err;
    }
}

CusparseVector & CusparseVector::operator=(const double val)
{
    //Make a local copy
    std::vector<double> vals(size);
	std::fill(vals.begin(), vals.end(), val);
    cudaError_t status = cudaMemcpy(d_v,&vals[0],sizeof(double)*size,cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf (stderr, "!!!! device access error (write A)\n");
        throw status;
    }
 
    //cudaMemcpy(&vals[0],d_v,size*sizeof(double),cudaMemcpyHostToDevice);
	return *this;
}

CusparseVector & CusparseVector::operator=(const std::vector<double> & vec)
{
    int size_ = vec.size();
    cudaError_t err;
    if(size!=size_)
    {
      err = cudaFree(d_v);
      err = cudaMalloc((void**)&d_v,size*sizeof(double));
      size = size_;
      if (err != cudaSuccess) {
        fprintf (stderr, "!!!! device access error (free Vector)\n");
        throw err;
      }
    }
    err = cudaMemcpy(d_v,&vec[0],sizeof(double)*size,cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf (stderr, "!!!! device access error (copy Vector)\n");
        throw err;
    }
 

    return *this;
}

CusparseVector & CusparseVector::operator=(const CusparseVector & vec)
{
    cudaError_t err;
   if(size!=vec.size)
    {
        size = vec.size;
        cudaFree(d_v);
        err = cudaMalloc(&d_v,sizeof(double)*size);
        if(err!=cudaSuccess)
        {
            fprintf (stderr, "cuda error, unable to make copy of array\n");
            throw err;
        }
        
    }
    err = cudaMemcpy(d_v,vec.d_v,sizeof(double)*size,cudaMemcpyDeviceToDevice);
    if(err!=cudaSuccess)
    {
        fprintf (stderr, "cuda error, unable to make copy of array\n");
        throw err;
    }
 
    return *this;
}


bool CusparseVector::operator==(const CusparseVector & RHS) const
{
    return RHS.d_v == d_v;
}

//! multiply THIS by a scalar value 
void CusparseVector::Scale(const double & val)
{
    //Since cusparse does not have a scale, lets use thrust
    thrust::transform(td_v,td_v+size,td_v,times(val));
}


//! Create a new vector with the same structure of THIS. Values are not initialized.
CusparseVector * CusparseVector::Clone()
{
	return new CusparseVector(handle,size);
}

double CusparseVector::operator[](const int i)
{
    //XXX: slow access!
	assert( i < size);
    double local_val; 
    cudaMemcpy(&local_val,d_v+i,sizeof(double),cudaMemcpyDeviceToHost);
	return local_val;
}

double CusparseVector::operator[](const int i) const
{
	assert( i < size );
    double local_val; 
    cudaMemcpy(&local_val,d_v+i,sizeof(double),cudaMemcpyDeviceToHost);
	return local_val;
}
 
double CusparseVector::at(const int i)
{

	if (i<0 || i > size-1)
		return 0.0;

	return operator[](i);
}

//Generate a random vector of unit norm in the host
//and copy to the device.
void CusparseVector::Randomize(int seed)
{
    std::vector<double> vals(size);

	srand(seed);
    double norm_sq = 0;
   
	for(int ix = 0; ix != size; ++ix)
    {
        vals[ix] = 2.*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1.;
        norm_sq += vals[ix]*vals[ix];
    }

    norm_sq = sqrt(norm_sq);
    for(int ix = 0; ix < size; ++ix)
    {
        vals[ix] = vals[ix]/norm_sq;
    }
    //TODO: check for errors 
    cudaMemcpy(&d_v[0],&vals[0],size*sizeof(double),cudaMemcpyHostToDevice);
}


void CusparseVector::Print(std::ostream & os) const
{
    std::vector<double> host_copy(size);
    //TODO: check for errors
    cudaError_t err = cudaMemcpy(&host_copy[0],d_v,sizeof(double)*size,cudaMemcpyDeviceToHost);
    //cudaMemcpy(d_v,&host_copy[0],size*sizeof(double),cudaMemcpyDeviceToHost);
	for(int ix = 0; ix != size; ++ix)
		os << host_copy[ix] << "\t ";
	os << "\n";
}


//! result = v1 + c2*v2
void add(const CusparseVector & v1, const double & c2, const CusparseVector & v2, CusparseVector & result)
{
    //TODO: Check for error
    thrust::device_ptr<double> res_ptr = result.td_v;
    thrust::device_ptr<double> a_ptr = v1.td_v;
    thrust::device_ptr<double> b_ptr = v2.td_v;

    int size = v1.size;
   
    //Execute the operation
    thrust::transform(a_ptr,a_ptr+size,b_ptr,res_ptr,daxpby(1., c2));
}

//! result = c1*v1 + c2*v2
void add(const double & c1, const CusparseVector & v1, const double & c2, const CusparseVector & v2, CusparseVector & result)
{
    //TODO: Check for error
    thrust::transform(v1.td_v,v1.td_v+v1.size,v2.td_v,result.td_v,daxpby(c1, c2));
}

//! result = alpha(v1 + v2)
void add(const double & alpha, const CusparseVector & v1, const CusparseVector & v2, CusparseVector & result)
{
    thrust::transform(v1.td_v,v1.td_v+v1.size,v2.td_v,result.td_v,daxpby(alpha, alpha));
}


//! result = v1 + v2 + v3
void add(const CusparseVector & v1, const CusparseVector & v2, const CusparseVector & v3, CusparseVector & result)
{

    //TODO: Check for errors
    const double one(1.);
    if(v3==result)
    {
        thrust::transform(v1.td_v,v1.td_v+v1.size,v3.td_v,result.td_v,daxpby(one, one));
        thrust::transform(v2.td_v,v2.td_v+v1.size,v3.td_v,result.td_v,daxpby(one, one));
    }
    else if(v2==result)
    {
        thrust::transform(v1.td_v,v1.td_v+v1.size,v2.td_v,result.td_v,daxpby(one, one));
        thrust::transform(v2.td_v,v2.td_v+v1.size,v3.td_v,result.td_v,daxpby(one, one)); 
    }
    else
    {
        thrust::transform(v1.td_v,v1.td_v+v1.size,v2.td_v,result.td_v,daxpby(one, one));
        thrust::transform(v1.td_v,v1.td_v+v1.size,v3.td_v,result.td_v,daxpby(one, one)); 
    }

}


//! result = v1 - v2
void subtract(const CusparseVector & v1, const CusparseVector & v2, CusparseVector & result)
{
    double minusone(-1.); 
    double one(1.);
    thrust::transform(v1.td_v,v1.td_v+v1.size,v2.td_v,result.td_v,daxpby(one, minusone));
}

//! return the inner product of v1 and v2
double InnerProduct(const CusparseVector & v1, const CusparseVector & v2)
{
    double zero(0.);
    return thrust::inner_product(v1.td_v,v1.td_v + v1.size,v2.td_v,zero);
}
