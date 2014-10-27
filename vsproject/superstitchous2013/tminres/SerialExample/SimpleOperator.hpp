/*
 * SimpleOperator.hpp
 *
 *  Created on: Sep 4, 2012
 *      Author: Santiago Akle
 */

#ifndef SIMPLEOPERATOR_HPP_
#define SIMPLEOPERATOR_HPP_

#define BAD_DATA_LENGTH 1
#include "SimpleVector.hpp"
#include <vector>
class SimpleOperator {
public:
	//! @class SimpleOperator
	/*!
	 * @brief  This class allows us to load an arbitrary hermitian dense symmetric
     * matrix and use cublas to form the product Ax
     * */
	//! Constructor
	/*!
	 * @param n_ int : number of cell in the x direction. The number of cell in the y direction is also n_.
	 */
	SimpleOperator(std::vector<double> v);
    SimpleOperator(int);
    ~SimpleOperator();
    //Copies the data int this
    SimpleOperator & operator=(const SimpleOperator & A);
    //Transforms into column major order and copies to the GPU
    SimpleOperator & operator=(std::vector<double> v);
    //transfer to local memory and print
    void Print(std::ostream & os); 
    //Fill with random values
    void Randomize(int seed);
	//! y = A*x
	void Apply(const SimpleVector & X, SimpleVector & y) const;

private:
    std::vector<double>* A;
	int n;
    
};

#endif /* CUDAOPERATOR_HPP_ */
