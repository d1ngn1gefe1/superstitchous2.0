/*
 * CudaVector.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: santiago akle
 *
 *  Modification from CudaVector.hpp by uvilla
 *
 */

#ifndef CUDAVECTOR_HPP_
#define CUDAVECTOR_HPP_

#include <cublas_v2.h>
#include <iostream>
#include <vector>

class CudaVector
{
public:
	//! @class CudaVector
	/*!
	 * @brief Implementation of a dense serial vector according to Vector_traits
	 */

	//! Constructor
	/*!
	 * @param size int : the size of the vector
	 */
	
    CudaVector(cublasHandle_t & handle, int n_);
	//! Destructor
	virtual ~CudaVector();

	//! Set all the entry of the Vector equal to val
	CudaVector &  operator=(const double & val);
	//! Set the entry of the Vector equal to the entries in RHS
	CudaVector & operator=(const CudaVector & RHS);
    //!Copy the values from the thrust::host_vector to the device vector
    CudaVector & operator=(const std::vector<double> & RHS);
    //!True if both vectors are the same object
    bool operator==(const CudaVector & RHS) const; 
    //! multiply THIS by a scalar value
	void Scale(const double & val);
	//! Create a new vector with the same structure of THIS. Values are not initialized.
	CudaVector * Clone();

	//! Access entry i (non const version)
	double operator[](const int i);
	//! Access entry i (const version)
	double operator[](const int i) const;
	//! Access entry i. if i < 0 return 0
	double at(const int i);

    //!Fill the vector with random numbers and normalize
    void Randomize(int seed);

    //! Print all the entries of the vector
    void Print(std::ostream & os) const;

	//! result = v1 + c2*v2
	friend void add(const CudaVector & v1, const double & c2, const CudaVector & v2, CudaVector & result);
	//! result = c1*v1 + c2*v2
	friend void add(const double & c1, const CudaVector & v1, const double & c2, const CudaVector & v2, CudaVector & result);
	//! result = alpha(v1 + v2)
	friend void add(const double & alpha, const CudaVector & v1, const CudaVector & v2, CudaVector & result);
	//! result = v1 + v2 + v3
	friend void add(const CudaVector & v1, const CudaVector & v2, const CudaVector & v3, CudaVector & result);
	//! result = v1 - v2
	friend void subtract(const CudaVector & v1, const CudaVector & v2, CudaVector & result);
	//! return the inner product of v1 and v2
	friend double InnerProduct(const CudaVector & v1, const CudaVector & v2);


private:
	double *d_v;
    int size;
    cublasHandle_t handle;
    cublasStatus_t status;

friend class CudaOperator;
};



#endif /* CUDAVECTOR_HPP_ */
