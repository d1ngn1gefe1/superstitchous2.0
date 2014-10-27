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
#include "CudaVector.hpp"
#include "CudaOperator.hpp"
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
    //Generate a problem 
    int n = 5;

    //Initialize the cublas context
    cublasHandle_t handle;
    //TODO: check errors
    cublasCreate(&handle);
    
    //Instantiate the appropriate Cuda stuctures
    CudaVector   c_rhs(handle,n);
    CudaOperator c_A(handle,n);
    
    //Transfer the data to the GPU
    c_rhs.Randomize(0);
    c_A.Randomize(0);
    
    //Call minres
    double shift(0);
	int max_iter(10000);
	double tol(1e-6);
	bool show(true);
    
    CudaVector x(handle,n);
    x = 0;

	//(8) Solve the problem with minres
	MINRES<CudaOperator,CudaVector,CudaOperator>(c_A, x, c_rhs, NULL, shift, max_iter, tol, show);

}

