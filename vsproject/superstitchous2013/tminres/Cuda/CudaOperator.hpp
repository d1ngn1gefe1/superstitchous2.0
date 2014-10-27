/*
 * CudaOperator.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Santiago Akle
 */

#ifndef CUDAOPERATOR_HPP_
#define CUDAOPERATOR_HPP_

#define BAD_DATA_LENGTH 1

#include "CudaVector.hpp"
#include <cublas_v2.h>
class CudaOperator {
public:
	//! @class CudaOperator
	/*!
	 * @brief  This class allows us to load an arbitrary hermitian dense symmetric
     * matrix and use cublas to form the product Ax
     * */

	//! Constructor
	/*!
	 * @param n_ int : number of cell in the x direction. The number of cell in the y direction is also n_.
	 */
	CudaOperator(cublasHandle_t handle, int n_, std::vector<double> v);
	CudaOperator(cublasHandle_t handle, std::vector<double> v);
    CudaOperator(cublasHandle_t handle_, int n_ );
    ~CudaOperator();
    //Copies the data int this
    CudaOperator & operator=(const CudaOperator & A);
    //Transforms into column major order and copies to the GPU
    CudaOperator & operator=(std::vector<double> v);
    //transfer to local memory and print
    void Print(std::ostream & os); 
    //Fill with random values
    void Randomize(int seed);
	//! y = A*x
	void Apply(const CudaVector & X, CudaVector & y) const;

private:
    double * d_A;
    cublasHandle_t handle;
    cublasStatus_t status;
	int n;
    
};

#endif /* CUDAOPERATOR_HPP_ */
