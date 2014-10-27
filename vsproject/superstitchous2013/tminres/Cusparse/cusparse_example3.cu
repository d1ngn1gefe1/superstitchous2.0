/*
 * cusparse_example3.cu
 * Created: September 16, 2012
 * Author: Santiago Akle
 *
 */
 
/* This example loads the matrix in matrix market format and solves 
 * using a random rhs.
 */
#include "tminres.hpp"
#include "CusparseVector.hpp"
#include "CusparseOperator.hpp"
#include <cmath>
#include <stdio.h>

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

    std::cout << "MINRES CUSPARSE with a symmetric matrix market matrix\n";

    //Open the marix market file
    FILE *mm_f = fopen("1138_bus.mtx","r");
    if(mm_f==NULL)
    {
        std::cerr << "Unable to open matrix file\n";
        throw -1;
    }

    //Define the cusparse opaque structures
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    
    //Load the operator
    CusparseOperator c_A = CusparseOperator::LoadMMFile(handle,mm_f);
    //Close the file
    fclose(mm_f);
  
    std::cout<<"Loaded matrix\n";

    //Instantiate the right hand side  
    int n = c_A.get_n();
    CusparseVector   c_rhs(handle,n);
    //Generate a random dense rhs
    std::vector<double> h_rhs(n);
    generate_random_values(h_rhs);
    //Copy to the CusparseVector object, which copies it into the GPU
    c_rhs = h_rhs; 

    std::cout<<"Generated rhs\n";

    double nsr = InnerProduct(c_rhs,c_rhs);
    std::cout << "norm of rhs: "<<nsr<<"\n";
    CusparseVector ab(handle,n);

    c_A.Apply(c_rhs,ab);
    std::cout << "test application\n";
    double nsab = InnerProduct(ab,ab);
    std::cout << "norm of Ab: "<<nsab<<"\n";

    //Call minres
    double shift(0);
	int max_iter(10000);
	double tol(1e-6);
	bool show(true);

    //Instantiate a holder for the solution 
    CusparseVector x(handle,n);
    x = 0;

	//(8) Solve the problem with minres
	MINRES<CusparseOperator,CusparseVector,CusparseOperator>(c_A, x, c_rhs, NULL, shift, max_iter, tol, show);


}

