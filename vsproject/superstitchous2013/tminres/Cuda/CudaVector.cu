/*
 * CudaVector.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: Santiago Akle
 */

#include "CudaVector.hpp"
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

CudaVector::CudaVector(cublasHandle_t & handle_, int size_):
	size(size_), handle(handle_)
{
    assert(size > 0);
    //Allocate the space in the GPU
    cudaError_t cudaStat = cudaMalloc((void**)&d_v,size*size*sizeof(double));
    if(cudaStat != cudaSuccess)
    {
       std::cerr << "Unable to allocate device memory for vector \n";
       throw cudaStat;
    }

}

CudaVector::~CudaVector()
{
    //TODO: error checking
	cudaFree(d_v);
}

CudaVector & CudaVector::operator=(const double & val)
{
    //Make a host copy of the data to fill the device vector with.
    std::vector<double> vals(size);
	std::fill(vals.begin(), vals.end(), val);
    cublasStatus_t status = cublasSetVector(size,sizeof(double),&vals[0],1,d_v,1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to copy vector to device! \n";
        throw status;
    } 
	return *this;
}


CudaVector & CudaVector::operator=(const std::vector<double> & vec)
{
    cublasStatus_t status = cublasSetVector(size,sizeof(double),&vec[0],1,d_v,1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to copy vector to device! \n";
        throw status;
    }
    return *this;
}

//! Set the entry of the Vector equal to the entries in RHS
CudaVector & CudaVector::operator=(const CudaVector & RHS)
{
    assert(size == RHS.size);
    //TODO: Error message for release mode
    cublasStatus_t status = cublasDcopy(handle,size,RHS.d_v,1,d_v,1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to copy data in the device! \n";
        throw status;
    }
	return *this;
}

bool CudaVector::operator==(const CudaVector & RHS) const
{
    return RHS.d_v == d_v;
}

//! multiply THIS by a scalar value
void CudaVector::Scale(const double & val)
{
    cublasStatus_t status = cublasDscal(handle, size, &val, d_v, 1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas operation failed \n";
        throw status;
    }
}

//! Create a new vector with the same structure of THIS. Values are not initialized.
CudaVector * CudaVector::Clone()
{
	return new CudaVector(handle,size);
}

double CudaVector::operator[](const int i)
{
    //XXX: slow access!
	assert( i < size);
    double local_val; 
    cudaError_t status = cudaMemcpy(&local_val,d_v+i,sizeof(double),cudaMemcpyDeviceToHost);
	if(status != cudaSuccess)
    {
        std::cerr << "Copy operation failed \n";
        throw status;
    }
    return local_val;
}

double CudaVector::operator[](const int i) const
{
	assert( i < size );
    double local_val; 
    cudaError_t status = cudaMemcpy(&local_val,d_v+i,sizeof(double),cudaMemcpyDeviceToHost);
	if(status != cudaSuccess)
    {
        std::cerr << "Copy operation failed \n";
        throw status;
    }
	return local_val;
}

double CudaVector::at(const int i)
{

	if (i<0 || i > size-1)
		return 0.0;

	return operator[](i);
}


void CudaVector::Randomize(int seed)
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
    cublasStatus_t status = cublasSetVector(size,sizeof(double),&vals[0],1,d_v,1);
	if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status;
    }
}

void CudaVector::Print(std::ostream & os) const
{
    std::vector<double> host_copy(size);
    //Copy to host 
    cublasStatus_t status = cublasGetVector(size,sizeof(double),d_v,1,&host_copy[0],1);
	if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status;
    }
	
    for(int ix = 0; ix != size; ++ix)
		os << host_copy[ix] << "\t ";
	os << "\n";
}

//These are not functions of the vector class, therefore they require access
//to a handle from one of the instances.

//! result = v1 + c2*v2
void add(const CudaVector & v1, const double & c2, const CudaVector & v2, CudaVector & result)
{
    cublasStatus_t status_c, status_d, status_s;
    status_c = CUBLAS_STATUS_SUCCESS;
    status_d = CUBLAS_STATUS_SUCCESS;
    status_s = CUBLAS_STATUS_SUCCESS;

    if(v2==result)
    {
        double one(1.);
        status_s = cublasDscal(result.handle,result.size,&c2,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v1.d_v,1,result.d_v,1);
    }
    else //The result and the parameter to scale are in different memory locations
    {
        status_c = cublasDcopy(result.handle,result.size,v1.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&c2,v2.d_v,1,result.d_v,1);
    }
    if(status_c != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status_c;
    }
    if(status_d != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas daxpy operation failed \n";
        throw status_d;
    }
    if(status_s != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas scale operation failed \n";
        throw status_s;
    }

}
//! result = c1*v1 + c2*v2
void add(const double & c1, const CudaVector & v1, const double & c2, const CudaVector & v2, CudaVector & result)
{
    cublasStatus_t status_c, status_d, status_s;
    status_c = CUBLAS_STATUS_SUCCESS;
    status_d = CUBLAS_STATUS_SUCCESS;
    status_s = CUBLAS_STATUS_SUCCESS;


    if(v2==result)
    {
        status_s = cublasDscal(result.handle,result.size,&c2,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&c1,v1.d_v,1,result.d_v,1);
    }
    else
    {
        status_c = cublasDcopy(result.handle,result.size,v1.d_v,1,result.d_v,1);
        status_s = cublasDscal(result.handle,result.size,&c1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&c2,v2.d_v,1,result.d_v,1);
    }
    if(status_c != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status_c;
    }
    if(status_d != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas daxpy operation failed \n";
        throw status_d;
    }
    if(status_s != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas scale operation failed \n";
        throw status_s;
    }


}

//! result = alpha(v1 + v2)
void add(const double & alpha, const CudaVector & v1, const CudaVector & v2, CudaVector & result)
{
    cublasStatus_t status_c, status_d, status_s;
    status_c = CUBLAS_STATUS_SUCCESS;
    status_d = CUBLAS_STATUS_SUCCESS;
    status_s = CUBLAS_STATUS_SUCCESS;


    const double one(1.);
    if(v2 == result)
    {
        status_d = cublasDaxpy(result.handle,result.size,&one,v1.d_v,1,result.d_v,1);
        status_s = cublasDscal(result.handle,result.size,&alpha,result.d_v,1);
    }
    status_c = cublasDcopy(result.handle,result.size,v1.d_v,1,result.d_v,1);
    status_d = cublasDaxpy(result.handle,result.size,&one,v2.d_v,1,result.d_v,1);
    status_s = cublasDscal(result.handle,result.size,&alpha,result.d_v,1);

    if(status_c != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status_c;
    }
    if(status_d != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas daxpy operation failed \n";
        throw status_d;
    }
    if(status_s != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas scale operation failed \n";
        throw status_s;
    }


}

//! result = v1 + v2 + v3
void add(const CudaVector & v1, const CudaVector & v2, const CudaVector & v3, CudaVector & result)
{
    cublasStatus_t status_c, status_d, status_s;
    status_c = CUBLAS_STATUS_SUCCESS;
    status_d = CUBLAS_STATUS_SUCCESS;
    status_s = CUBLAS_STATUS_SUCCESS;

    const double one(1.);
    if(v3==result)
    {
        status_d = cublasDaxpy(result.handle,result.size,&one,v2.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v1.d_v,1,result.d_v,1);
    
    }
    else if(v2==result)
    {
        status_d = cublasDaxpy(result.handle,result.size,&one,v3.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v1.d_v,1,result.d_v,1);
    }
    else
    {
        status_c = cublasDcopy(result.handle,result.size,v1.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v2.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v3.d_v,1,result.d_v,1);
    }

    if(status_c != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status_c;
    }
    if(status_d != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas daxpy operation failed \n";
        throw status_d;
    }
    if(status_s != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas scale operation failed \n";
        throw status_s;
    }


}

//! result = v1 - v2
void subtract(const CudaVector & v1, const CudaVector & v2, CudaVector & result)
{
    cublasStatus_t status_c, status_d, status_s;
    status_c = CUBLAS_STATUS_SUCCESS;
    status_d = CUBLAS_STATUS_SUCCESS;
    status_s = CUBLAS_STATUS_SUCCESS;

    const double one(-1.);
    if(v2==result)
    {
        status_d = cublasDaxpy(result.handle,result.size,&one,v1.d_v,1,result.d_v,1);  
        status_s = cublasDscal(result.handle,result.size,&one,result.d_v,1);
    }
    else
    {
        status_c = cublasDcopy(result.handle,result.size,v1.d_v,1,result.d_v,1);
        status_d = cublasDaxpy(result.handle,result.size,&one,v2.d_v,1,result.d_v,1); 
    }

    if(status_c != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Copy operation failed \n";
        throw status_c;
    }
    if(status_d != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas daxpy operation failed \n";
        throw status_d;
    }
    if(status_s != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas scale operation failed \n";
        throw status_s;
    }


}

//! return the inner product of v1 and v2
double InnerProduct(const CudaVector & v1, const CudaVector & v2)
{
    double res = 0;
    cublasStatus_t status = cublasDdot(v1.handle,v1.size,v1.d_v,1,v2.d_v,1,&res);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas inner product operation failed \n";
        throw status;
    }
    return res;
}

