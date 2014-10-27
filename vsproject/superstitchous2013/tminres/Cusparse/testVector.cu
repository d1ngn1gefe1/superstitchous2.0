#include "CusparseVector.hpp"
#include "CusparseSymmetricOperator.hpp"
#include <vector>

/*
 * This code checks the implementation of the methods in SimpleVector.
 */
int main()
{

    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
       std::cerr << "Unable to initialize cusparse\n";
       throw status;
    }
    
    //Initialize the cusparse
	int dim(6);
	CusparseVector a(handle,dim), b(handle,dim), c(handle,dim), d(handle,dim);
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

   	std::cout << "InnerProduct(b,d) = " << InnerProduct(b,d) << "\n";

	std::cout << "InnerProduct(d,d) = " << InnerProduct(d,d) << "\n";

    //Since the matrix is symmetric we only store the upper triangular section
    //the format is one based crs
    double vals_a[] = {1.,2.,3.,1.,1.,4.,1.,1.,5.,1.};
    int col_ix_a[]    = {1,3,5,2,3,4,4,5,6,6};
    int row_ptr_a[]   = {1,4,5,7,8,10,11};  
    
    std::vector<double> values(vals_a,vals_a + 10);
    std::vector<int>    col_ix(col_ix_a,col_ix_a + 10);
    std::vector<int>    row_ptr(row_ptr_a,row_ptr_a+7);
    
    //Define the operator 
    CusparseSymmetricOperator op(handle,row_ptr,col_ix,values);
    std::cout << "Defined the Symmetric operator\n";
    //Execute the operation
    op.Apply(a,b); 
    std::cout << "Executed the product\n";
    b.Print(std::cout);

	return 0;
}
