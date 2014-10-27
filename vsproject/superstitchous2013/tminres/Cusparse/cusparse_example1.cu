/*
 * cuda_example1.cpp
 * Created: Jun 27, 2012
 * Author: Santiago Akle
 *
 */
 
/* This example generates a random problem 
 * A,b and solves it using Umberto Villas C minres and CUDA.
 */

#include "tminres.hpp"
#include "CusparseVector.hpp"
#include "CusparseOperator.hpp"
#include <cmath>

class rand_functor
{
public:
    rand_functor(int seed)
    {
        srand(seed);
    }
    double operator()()
    {
        return rand()/(double)RAND_MAX;
    }
};

void generate_random_values(std::vector<double> &d_a)
{

    std::generate(d_a.begin(),d_a.end(),rand_functor(0));
}


int main()
{

    std::cout << "MINRES CUSPARSE with a symmetric operator in a general matrix\n";

    //Generate a sparse symmetric problem
    int n = 6;
    int nnz = 12;

    //Define the cusparse opaque structures
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
 
    //Instantiate the right hand side 
    CusparseVector   c_rhs(handle,n);
    
    //Since this code uses the general operator we need to encode all the entries of
    // the symmetric matrix.

    //Define a prescribed sparsity pattern with random values
    int col_ix_a[]    = {1,2,3,1,2,1,3,4,3,4,5,6};
    int row_ptr_a[]   = {1,4,6,9,11,12,13};    
    
    std::vector<double> r(9);
    generate_random_values(r);
    //Put the values in the correct order for the matrix to be symmetric
    double values_a[]   = {r[0],r[1],r[2],r[1],r[3],r[2],r[4],r[5],r[5],r[6],r[7],r[8]};
    std::vector<double> values(values_a,values_a+nnz);
    
    std::vector<int>    col_ix(col_ix_a,col_ix_a + nnz);
    std::vector<int>    row_ptr(row_ptr_a,row_ptr_a+n+1); 
    CusparseOperator c_A(handle,row_ptr,col_ix,values);

    //Generate a random dense rhs
    //double rhs_vals_a[] = {0,0,0,1,1,1};
    //std::vector<double> h_rhs(rhs_vals_a,rhs_vals_a + 6);
    std::vector<double> h_rhs(6);
    generate_random_values(h_rhs);
    //Copy to the gpu
    c_rhs = h_rhs; 

    //Call minres
    double shift(0);
	int max_iter(10000);
	double tol(1e-6);
	bool show(true);
    
    CusparseVector x(handle,n);
    x = 0;

	//(8) Solve the problem with minres
	MINRES<CusparseOperator,CusparseVector,CusparseOperator>(c_A, x, c_rhs, NULL, shift, max_iter, tol, show);

}

