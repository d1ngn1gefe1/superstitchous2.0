/*
 * CusparseSymmetricOperator.cpp
 * Created on: Sep 16, 2012
 *  Author: Santiago Akle
 *
 */

#include "CusparseSymmetricOperator.hpp"
#include "cusparse_v2.h"
#include <cuda_runtime.h>
#include <vector>

CusparseSymmetricOperator::CusparseSymmetricOperator(cusparseHandle_t handle_, std::vector<int> row_ptr, std::vector<int> col_ix, std::vector<double> vals): CusparseOperator(handle_)
{
   cudaError_t cudaStat;
   //Set the number of non zeros
   nnz = vals.size(); 
   //Set the number of rows
   n   = row_ptr.size()-1;

   cusparseStatus_t err = cusparseCreateMatDescr(&matDesc); 
   if(err!=CUSPARSE_STATUS_SUCCESS)
   {
      std::cerr << "Unable to allocate matrix descriptor\n";
      throw err;

   }
    
   //Set the type to hermitian and the index base to one 
   cusparseSetMatType (matDesc, CUSPARSE_MATRIX_TYPE_SYMMETRIC);  
   cusparseSetMatIndexBase (matDesc, CUSPARSE_INDEX_BASE_ONE) ;
   //Assume that the data corresponds to the upper triangular section of 
   //the matrix.
   cusparseSetMatFillMode(matDesc, CUSPARSE_FILL_MODE_UPPER);

   // Allocate the space for the vectors that define the matrix 
   cudaStat = cudaMalloc((void**)&csrValA,sizeof(double)*nnz);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
  // Space for the values
   cudaStat = cudaMalloc((void**)&csrColIndA,sizeof(int)*nnz);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
   //Space for the pointers to the row starts
   cudaStat = cudaMalloc((void**)&csrRowPtrA,sizeof(int)*(n+1));
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }

  //Load the column indices to the gpu
   cudaStat = cudaMemcpy(csrColIndA,&col_ix[0],sizeof(int)*col_ix.size(),cudaMemcpyHostToDevice);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
 
   //Load the matrix values to the gpu
   cudaStat = cudaMemcpy(csrValA,&vals[0],sizeof(double)*vals.size(),cudaMemcpyHostToDevice); 
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }

   //Load the row start poitners to the gpu
   cudaStat = cudaMemcpy(csrRowPtrA,&row_ptr[0],sizeof(int)*row_ptr.size(),cudaMemcpyHostToDevice); 
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
   
}

