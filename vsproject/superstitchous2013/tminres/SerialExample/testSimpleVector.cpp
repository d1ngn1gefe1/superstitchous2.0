// tminres is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
// Authors:
// - Umberto Villa, Emory University - uvilla@emory.edu
// - Michael Saunders, Stanford University
// - Santiago Akle, Stanford University

/*!
@file
@author U. Villa - uvilla@emory.edu
@date 04/2012
*/


#include "SimpleVector.hpp"

/*
 * This code checks the implementation of the methods in SimpleVector.
 */
int main()
{
	int dim(5);
	SimpleVector a(dim), b(dim), c(dim), d(dim);
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

	subtract(d,b, a);
	std::cout << "a = d - b = ";
	a.Print(std::cout);

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

	return 0;
}
