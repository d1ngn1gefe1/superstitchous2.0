/*
 * CusparseVector.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: santiago akle
 *
 *  Modification from CusparseVector.hpp by uvilla
 *
 */

#ifndef CUSPARSEVECTOR_HPP_
#define CUSPARSEVECTOR_HPP_

#include <cusparse_v2.h>
#include <iostream>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>

class CusparseVector
{
public:

    //! @class CusparseVector
	/*!
	 * @brief Implementation of a dense serial vector according to Vector_traits
	 */

	//! Constructor
	/*!
	 * @param size int : the size of the vector
	 */
	
    CusparseVector(cusparseHandle_t & handle,int n_);
	//! Destructor
	virtual ~CusparseVector();

    //!Copy the values from the std::host_vector to the device vector
    //and make a dense vector
    CusparseVector & operator=(const std::vector<double> & RHS);
    //Fill vectpr with one value
    CusparseVector & operator=(const double val);
    //Copy the values in the GPU
    CusparseVector & operator=(const CusparseVector & vec);
    
    //!True if both vectors are the same object
    bool operator==(const CusparseVector & RHS) const; 
    //! multiply THIS by a scalar value
	void Scale(const double & val);
	//! Create a new vector with the same structure of THIS. Values are not initialized.
	CusparseVector * Clone();

	//! Access entry i (non const version)
	double operator[](const int i);
	//! Access entry i (const version)
	double operator[](const int i) const;
	//! Access entry i. if i < 0 return 0
	double at(const int i);

    //! Copies the data into the device, must be a sparse vector with cols_ix.size() == nnz
    void Load(std::vector<double> &cols_ix, std::vector<double> &values);    
    //Generates a random vector of unit length
    void Randomize(int seed);
    //Copies to host memory and prints in std out
    void Print(std::ostream & os) const;
	//! result = v1 + c2*v2
	friend void add(const CusparseVector & v1, const double & c2, const CusparseVector & v2, CusparseVector & result);
	//! result = c1*v1 + c2*v2
	friend void add(const double & c1, const CusparseVector & v1, const double & c2, const CusparseVector & v2, CusparseVector & result);
	//! result = alpha(v1 + v2)
	friend void add(const double & alpha, const CusparseVector & v1, const CusparseVector & v2, CusparseVector & result);
	//! result = v1 + v2 + v3
	friend void add(const CusparseVector & v1, const CusparseVector & v2, const CusparseVector & v3, CusparseVector & result);
	//! result = v1 - v2
	friend void subtract(const CusparseVector & v1, const CusparseVector & v2, CusparseVector & result);
	//! return the inner product of v1 and v2
	friend double InnerProduct(const CusparseVector & v1, const CusparseVector & v2);


private:
	double *d_v;
    thrust::device_ptr<double> td_v;
    int size;
    cusparseHandle_t handle;

friend class CusparseOperator;
};

//Functor to scale a vector by a scalar
struct times : public thrust::unary_function<double,double>
  {
    double alpha;
    __host__ __device__
    times(double alpha_):alpha(alpha_)
    {}
    __host__ __device__
    double operator()(double x) { return alpha*x; }
  };

//Functor for daxpy
// y <-a*x+b*y
struct daxpby : public thrust::binary_function<double,double,double>
  {
    double alpha;
    double beta;

    __host__ __device__
    daxpby(double alpha_,double beta_):alpha(alpha_),beta(beta_)
    {}

    __host__ __device__
    double operator()(double x,double y) { return alpha*x+beta*y; }
  };


#endif /* CUSPARSEVECTOR_HPP_ */
