/*
 * CudaOperator.cpp
 * Created on: Jun 26, 2012
 *  Author: Santiago Akle
 *
 */

#include "CudaOperator.hpp"
#include <vector>

//Constructor for an empty matrix
CudaOperator::CudaOperator(cublasHandle_t handle_, int n_ ): n(n_), handle(handle_)
{
    //allocate space for the matrix
    cudaError_t cudaStat = cudaMalloc((void**)&d_A,sizeof(double)*n*n);
    if(cudaStat != cudaSuccess)
    {
       std::cerr << "Unable to allocate device memory for operator\n";
       throw cudaStat;
    }
}

//Constructor which receives a vector with the data 
CudaOperator::CudaOperator(cublasHandle_t handle_, int n_, std::vector<double> dat): n(n_), handle(handle_)
{
    //allocate space for the matrix
    cudaError_t cudaStat = cudaMalloc((void**)&d_A,sizeof(double)*n*n);
    if(cudaStat != cudaSuccess)
    {
       std::cerr << "Unable to allocate device memory for operator\n";
       throw cudaStat;
    }
    *this=dat;
    //assuming the matrix is in row major order it has to be reordered before transfer
}

//Constructor which receives a vector with the data 
CudaOperator::CudaOperator(cublasHandle_t handle_, std::vector<double> dat): handle(handle_)
{
    n = sqrt(dat.size());
    if(n*n != dat.size())
    {
        std::cerr << "Data vector length is not a square number\n";
        throw BAD_DATA_LENGTH;
    }
    
    //allocate space for the matrix
    cudaError_t cudaStat = cudaMalloc((void**)&d_A,sizeof(double)*n*n);
    if(cudaStat != cudaSuccess)
    {
       std::cerr << "Unable to allocate device memory for operator\n";
       throw cudaStat;
    }
    *this=dat;
    //assuming the matrix is in row major order it has to be reordered before transfer
}


CudaOperator::~CudaOperator()
{
    cudaFree(d_A);
}


void CudaOperator::Apply(const CudaVector & x, CudaVector & y) const
{
    const double alpha = 1;
    const double beta  = 0;
    cublasStatus_t status = cublasDsymv(handle,CUBLAS_FILL_MODE_UPPER,n,&alpha,d_A,n,x.d_v,1,&beta,y.d_v,1);  
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Error while executing the product \n";
        throw status;
    }
}

CudaOperator & CudaOperator::operator=(const CudaOperator & A)
{
    status = cublasDcopy(handle,n,A.d_A,1,d_A,1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to make a copy of the matrix in the device \n";
        throw status;
    }
    return *this;
}

//Copies the data into the device, assumes that the vector is in 
//row-major order and transposes.
CudaOperator & CudaOperator::operator=(std::vector<double> vec)
{
    if(vec.size() != n*n)
    {
        std::cerr << "Matrix and data sizes do not match \n";
        throw -1;
    }
    std::vector<double> transpose(n*n);
    for(int i=0;i<n;++i)
      for (int j = 0;j<n;++j)
        transpose[i+j*n] = vec[j+i*n];
    //Copy to the GPU
    status = cublasSetMatrix(n,n,sizeof(double),&transpose[0],n,d_A,n);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to transfer matrix to device \n";
        throw status;
    }
    return *this;
}


void CudaOperator::Randomize(int seed)
{
    std::vector<double> vals(n*n);
	srand(seed);
	for(int ix = 0; ix != n*n; ++ix)
    {
        vals[ix] = 2.*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1.;
    }

    status = cublasSetMatrix(n,n,sizeof(double),&vals[0],n,d_A,n);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to copy matrix to device \n";
        throw status;
    }

}


void CudaOperator::Print(std::ostream & os)
{
    std::vector<double> local_copy(n*n);
    status = cublasGetMatrix(n,n,sizeof(double),d_A,n,&local_copy[0],n);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Unable to copy matrix to device \n";
        throw status;
    }

    os << "\n";
    for(int i = 0; i<n;++i)
    {

        for(int j=0;j<n;++j)
        {
            os << local_copy[n*j+i] << "\t";
        }

        os << "\n";
    }

}
