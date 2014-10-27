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
#include "CusparseSymmetricOperator.hpp"
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
    std::cout << "MINRES CUSPARSE with a symmetric operator \n";

    //Generate a problem 
    int n = 6;
    int nnz = 9;

    //Define the cusparse opaque structures
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
 
 
    //Instantiate the right hand side 
    CusparseVector   c_rhs(handle,n);
    
    //Since the matrix is symmetric we only store the upper triangular section
    //for this example the sparsity pattern will be hard coded but the 
    //entries will be random 
    int col_ix_a[]    = {1,2,3,2,3,4,4,5,6};
    int row_ptr_a[]   = {1,4,5,7,8,9,10};   
    //Generate the values for the upper triangular part
    std::vector<double> values(nnz);
    generate_random_values(values);
    std::vector<int>    col_ix(col_ix_a,col_ix_a + nnz);
    std::vector<int>    row_ptr(row_ptr_a,row_ptr_a+n+1); 
    CusparseSymmetricOperator c_A(handle,row_ptr,col_ix,values);

    //Generate a random dense rhs
    //double rhs_vals_a[] = {0,0,0,1,1,1};
    //std::vector<double> h_rhs(rhs_vals_a,rhs_vals_a + 6);
    std::vector<double> h_rhs(6);
    generate_random_values(h_rhs);
    //Copy to the gpu
    c_rhs = h_rhs; 


    CusparseVector test(handle,n);
    c_A.Apply(c_rhs,test);
    test.Print(std::cout);

    //Call minres
    double shift(0);
	int max_iter(10000);
	double tol(1e-6);
	bool show(true);
    
    CusparseVector x(handle,n);
    x = 0;

	//(8) Solve the problem with minres
	MINRES<CusparseSymmetricOperator,CusparseVector,CusparseSymmetricOperator>(c_A, x, c_rhs, NULL, shift, max_iter, tol, show);

}

