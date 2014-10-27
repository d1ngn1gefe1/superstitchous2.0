/*
 * testSimpleVector.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: uvilla
 */


#include "CudaVector.hpp"
#include "CudaOperator.hpp"
#include <vector>

/*
 * This code checks the implementation of the methods in SimpleVector.
 */
int main()
{

    cublasStatus_t status;  
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    //Initialize the cublas 
	int dim(5);
	CudaVector a(handle,dim), b(handle,dim), c(handle,dim), d(handle,dim);
	a = 1.0;
	std::cout<< "a = ";
	a.Print(std::cout);

	b.Randomize(1);
	std::cout<< "b = ";
	b.Print(std::cout);

	c = b;
	std::cout<< "c = b =";
	c.Print(std::cout);

	c.Scale(-1.);
	std::cout << "c * (-1.) = ";
	c.Print(std::cout);

	d.Randomize(2);
	std::cout<<"d = ";
	d.Print(std::cout);

	subtract(d, a, a);
	std::cout << "a = d - a = ";
	a.Print(std::cout);

	subtract(d, a, c);
	std::cout << "c = d - a = ";
	c.Print(std::cout);

	add(d, 5., b, a);
	std::cout<< " a = d + 5.*b = ";
	a.Print(std::cout);

	add(0.5, d, 2.5, b, a);
	std::cout<< "a = 0.5 * d + 2.5*b = ";
	a.Print(std::cout);

	add(2., d, b, a);
	std::cout<< "a = 2* (d + b) = ";
	a.Print(std::cout);

	add(b,d,c, a);
	std::cout << "a = b + c + d = ";
	a.Print(std::cout);

    CudaOperator op(handle,dim);
    op.Randomize(0);
    std::cout << "Operator "; 
    op.Print(std::cout);

    op.Apply(a,a);
    std::cout << "Op*a ";
    a.Print(std::cout);    

	std::cout << "InnerProduct(b,d) = " << InnerProduct(b,d) << "\n";

	return 0;
}
