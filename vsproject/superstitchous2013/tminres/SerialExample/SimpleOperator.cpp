/*
 * SimpleOperator.cpp
 * Created on: Jun 26, 2012
 *  Author: Santiago Akle
 *
 */

#include "SimpleOperator.hpp"
#include <vector>
#include <math.h>
#include <cstdlib>

//Constructor for an empty matrix
SimpleOperator::SimpleOperator(int n_ ):n(n_)
{
    A = new std::vector<double>(n*n);
}

//Constructor which receives a vector with the data 
SimpleOperator::SimpleOperator(std::vector<double> dat)
{

    n = sqrt(dat.size());
    if(n*n != dat.size())
    {
        std::cerr << "Data vector length is not a square number\n";
        throw BAD_DATA_LENGTH;
    }
    std::copy(dat.begin(),dat.end(),A->begin());
}

SimpleOperator::~SimpleOperator()
{
   delete A;
}

//Calculates the product using the upper triangular section of the matrix
void SimpleOperator::Apply(const SimpleVector & x, SimpleVector & y) const
{
    double res;
    for(int i = 0;i<n;++i)
    {
        res = 0;
        int j;
        for(j=i;j<n;j++)
        {
            res+= x[j]*(*A)[n*i+j];
        }
        for(j=0;j<i;++j)
        {
            res+= x[j]*(*A)[n*j+i];
        }
        y[i] = res;

    }
}

SimpleOperator & SimpleOperator::operator=(const SimpleOperator & O)
{
    std::copy(O.A->begin(),O.A->end(),A->begin());
   
    return *this;
}

//Copies the data into the device, assumes that the vector is in 
//row-major order and transposes.
SimpleOperator & SimpleOperator::operator=(std::vector<double> vec)
{
    if(vec.size() != n*n)
    {
        std::cerr << "Matrix and data sizes do not match \n";
        throw -1;
    }
    std::copy(vec.begin(),vec.end(),A->begin());
    return *this;
}


void SimpleOperator::Randomize(int seed)
{
	srand(seed);
	for(int ix = 0; ix != n*n; ++ix)
    {
        (*A)[ix] = 2.*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1.;
    }
}


void SimpleOperator::Print(std::ostream & os)
{

    os << "\n";
    for(int i = 0; i<n;++i)
    {

        for(int j=0;j<n;++j)
        {
            os << (*A)[n*i+j] << "\t";
        }

        os << "\n";
    }

}
