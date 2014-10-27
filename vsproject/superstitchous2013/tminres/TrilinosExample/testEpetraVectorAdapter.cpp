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

/*
 * This code checks the implementation of the methods in EpetraVectorAdapter.
 *
 * mpirun -n 2 xterm -e " ./testEpetraVectorAdapter.exe; read -p 'Press enter to close window' "
 */


#include "EpetraVectorAdapter.hpp"

#include <Epetra_ConfigDefs.h>
#include <Epetra_MpiComm.h>
#include <mpi.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

int main(int argc,char * argv[])
{

	MPI_Init(&argc, &argv);
	std::cout<< "MPI Initialization\n";

	Epetra_MpiComm * comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	{
	int myPid(comm->MyPID());
	int nProc(comm->NumProc());

	int nEl(5*nProc);
	Epetra_Map map(nEl, 0, *comm);

	Epetra_Vector aev(map), bev(map), cev(map), dev(map);

	EpetraVectorAdapter a(aev), b(bev), c(cev), d(dev);
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

	}
	delete comm;
	MPI_Finalize();

	return 0;
}
