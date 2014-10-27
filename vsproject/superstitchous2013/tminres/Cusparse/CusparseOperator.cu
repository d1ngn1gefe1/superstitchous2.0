/*
 * CusparseOperator.cpp
 * Created on: Jun 26, 2012
 *  Author: Santiago Akle
 *
 */

#include "CusparseOperator.hpp"
#include "cusparse_v2.h"
#include <cuda_runtime.h>
#include <vector>
#include <mmio.h>
#include <stdio.h>

CusparseOperator::CusparseOperator(cusparseHandle_t handle_, std::vector<int> row_ptr, std::vector<int> col_ix, std::vector<double> vals): handle(handle_)
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
   cusparseSetMatType (matDesc, CUSPARSE_MATRIX_TYPE_GENERAL);  
   cusparseSetMatIndexBase (matDesc, CUSPARSE_INDEX_BASE_ONE) ;

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

CusparseOperator::CusparseOperator(cusparseHandle_t handle_,cusparseMatDescr_t matDesc_ ,double* csrValA_, int* csrColIndA_, int* csrRowPtrA_, int n_, int nnz_):
    handle(handle_),
    matDesc(matDesc_),
    csrValA(csrValA_),
    csrRowPtrA(csrRowPtrA_),
    csrColIndA(csrColIndA_),
    n(n_),
    nnz(nnz_)
{
   if(handle==NULL||matDesc==NULL||csrValA==NULL||csrColIndA==NULL||csrRowPtrA==NULL)
   {
      std::cerr << "Matrix initialized with null pointer\n";
      throw -1;
   }
}

CusparseOperator::CusparseOperator(cusparseHandle_t handle_): handle(handle_)
{}

CusparseOperator::~CusparseOperator()
{
    cudaFree(csrValA);
    cudaFree(csrColIndA);
    cudaFree(csrRowPtrA);
    cusparseStatus_t cusparseStatus = cusparseDestroyMatDescr(matDesc);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
       std::cerr << "Unable to free descriptor\n";
       throw cusparseStatus;
    }
}

int CusparseOperator::get_n()
{
    return n;
}

int CusparseOperator::get_nnz()
{
    return nnz;
}


//Loads file
CusparseOperator CusparseOperator::LoadMMFile(cusparseHandle_t handle_, FILE* mm_f)
{

   int m,n,nnz;
   MM_typecode mm_t;
   cudaError_t cudaStat;
   cusparseMatDescr_t matDesc;   

   //Pointers to the device arrays 
   double* csrValA;
   int*    csrColIndA;
   int*    csrRowPtrA;

   //Initialize the matrix descriptor
   cusparseStatus_t err = cusparseCreateMatDescr(&matDesc); 
   if(err!=CUSPARSE_STATUS_SUCCESS)
   {
      std::cerr << "Unable to allocate matrix descriptor\n";
      throw err;

   } 

   //Read the matrix description form the file
   int mm_io_err = mm_read_banner(mm_f,&mm_t);
   if(mm_io_err!=0)
   {
       std::cerr << "Error while reading matrix market file \n";
       throw mm_io_err;
   }
   if(mm_is_complex(mm_t))
   {

       std::cerr << "Complex matrices are not supported \n";
       throw TMINRES_CUBLAS_MM_UNSUPORTED;
   }
   if(!mm_is_sparse(mm_t))
   {
        std::cerr << "Dense matrices are not supported by this operator \n";
        throw TMINRES_CUBLAS_MM_UNSUPORTED;
   }
   if(!mm_is_matrix(mm_t))
   {
        std::cerr << "File must be a matrix to be used by this operator \n";
        throw TMINRES_CUBLAS_MM_UNSUPORTED; 
   }
   if(!mm_is_symmetric(mm_t))
   {
        std::cerr << "Matrix is not symmetric \n";
        throw TMINRES_CUBLAS_MM_UNSUPORTED; 
   }
   //Set the type to hermitian and the fill mode to upper
   cusparseSetMatType (matDesc, CUSPARSE_MATRIX_TYPE_SYMMETRIC);  
   cusparseSetMatFillMode(matDesc, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatIndexBase (matDesc, CUSPARSE_INDEX_BASE_ONE);
   
   //This call will skip the comments and then read the size of the matrix
   int ret_code = mm_read_mtx_crd_size(mm_f,&m,&n,&nnz);
   if (ret_code !=0)
   {
     std::cerr << "Unable to read file size \n"; 
     throw TMINRES_CUBLAS_MM_UNSUPORTED;  
   }

   //Validate that the matrix is square  
   if(m!=n)
   {
     std::cerr << "Matrix is not square \n";
     throw TMINRES_CUBLAS_MM_UNSUPORTED;   
   } 

   //Matrix market sparse files are stored in COO format so
   //we need to transform the row_ix to csr
   std::vector<double> values(nnz);
   std::vector<int>    col_ix(nnz);
   std::vector<int>    row_ix(nnz);

   //Matrix market sparse file store the lower triangular part
   // in column major order. We require row major order to form 
   // the csr format, therefore we will interpret the column coordinates
   // as row coordinates and the row as column coordinates and assume that the
   // upper triangular section is stored. 
 
   int i;
   for(i = 0; i < nnz; ++i)
      fscanf(mm_f, "%d %d %lg\n", &col_ix[i], &row_ix[i], &values[i]);
 
   //Pointer to the row coordinates in the device
   int* cooRowIndA; 

   // Space for the row coordinates
   cudaStat = cudaMalloc((void**)&cooRowIndA,sizeof(int)*(nnz));
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
   //Space for the row pointers
   cudaStat = cudaMalloc((void**)&csrRowPtrA,sizeof(int)*(n+1));
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
   //Copy the rows and convert to CRS
   cudaStat = cudaMemcpy(cooRowIndA,&row_ix[0],sizeof(int)*row_ix.size(),cudaMemcpyHostToDevice);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to copy data to device memory\n";
      throw cudaStat;
   }
   //Call the conversion routine
   cusparseStatus_t staus = cusparseXcoo2csr(handle_,cooRowIndA,nnz,n,csrRowPtrA,CUSPARSE_INDEX_BASE_ONE);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to covert to CSR format\n";
      throw cudaStat;
   }
    
   //Wait for the async call to finish
   cudaStat = cudaDeviceSynchronize();
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
 
    // Space for the column coordinates
   cudaStat = cudaMalloc((void**)&csrColIndA,sizeof(int)*nnz);
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
 
   //Free the space used for the row coordinates
   cudaStat = cudaFree(cooRowIndA);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to free memory from the coo format conversion will proceed\n";
   }


   //Copy the values
   // Allocate the space for the vectors that define the matrix 
   cudaStat = cudaMalloc((void**)&csrValA,sizeof(double)*nnz);
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }

   //Load the matrix values to the gpu
   cudaStat = cudaMemcpy(csrValA,&values[0],sizeof(double)*values.size(),cudaMemcpyHostToDevice); 
   if(cudaStat != cudaSuccess)
   {
      std::cerr << "Unable to allocate device memory for operator\n";
      throw cudaStat;
   }
 
   //Instantiate the object and pass the parameters
   return CusparseOperator(handle_, matDesc,csrValA,csrColIndA,csrRowPtrA,n,nnz);
}

void CusparseOperator::Apply(const CusparseVector & x, CusparseVector & y) const
{
    const double alpha = 1;
    const double beta  = 0;
    cusparseStatus_t cusparseStatus = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,n,nnz,&alpha,matDesc,csrValA,csrRowPtrA,csrColIndA,x.d_v,&beta,y.d_v);  
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
       std::cerr << "Unable to execute matrix vector product, Error: "<< cusparseStatus<<"\n";
       throw cusparseStatus;
    }
}
