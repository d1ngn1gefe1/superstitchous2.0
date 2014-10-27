/*
 * CusparseOperator.cu
 *
 *  Created on: Aug 14, 2012
 *      Author: Santiago Akle
 */

#ifndef CUSPARSESYMMETRICOPERATOR_HPP_
#define CUSPARSESYMMETRICOPERATOR_HPP_

#include "CusparseVector.hpp"
#include "CusparseOperator.hpp"
#include <cusparse_v2.h>

class CusparseSymmetricOperator: public CusparseOperator {
public:
	//! @class CusparseOperator
	/*!
	 * @brief  This class represents sparse symmetric operators stored in CSR format.
     * It is assumed that only the upper triangular part and its coordinates are stored.
     * It implements the call to NVIDIA CUSPARSE sparse symmetric product.
     * */

	//! Constructor
	/*!
	 * @param handle_ : Cusparse library handle 
     * @param row_ptr : Vector of length n+1 with the indices of the row starts plus nnz+1 (in the upper triangular part)
     * @param col_ix  : Vector of length nnz with the column coordinates of the non zero values of the upper triangular part
     * @param vals    : Vector of length nnz with the values of the upper triangular part of the matrix.
	 */
	
    CusparseSymmetricOperator(cusparseHandle_t handle_, std::vector<int> row_ptr, std::vector<int> col_ix, std::vector<double> vals);

};
#endif /* CUSPARSEOPERATOR_HPP_ */

