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

#include "EpetraVectorAdapter.hpp"
#include "EpetraOperatorAdapter.hpp"
#include "tminres.hpp"

#include <mpi.h>

#include <Epetra_ConfigDefs.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <ml_MultiLevelPreconditioner.h>
#include <Teuchos_ParameterList.hpp>

/*!
 * @example TrilinosExample/ex1.cpp
 *
 * This code shows how to use MINRES() with Epetra_Vector and Epetra_Matrices from Trilinos.
 *
 * We solve a 3D generalized Stokes problem in the unitary cube:
 *
 * @f[ u - \Delta u + \nabla p = f @f]
 * @f[ {\rm div}  u            = 0 @f]
 *
 * where u is the velocity field and p the scalar pressure fields.
 *
 *
 * We use Tailor-Hood element (P2-P1) finite elements for the discretization of the pressure and velocity field.
 * The discrete saddle point system reads
 *
 * @f[
 * \left[ \begin{array}{cc} C & B^T \\ B & 0 \end{array} \right]
 * \left[ \begin{array}{c}  u \\ p \end{array} \right] =
 * \left[ \begin{array}{c}  f \\ 0 \end{array} \right]
 * @f]
 *
 * where C = M + A is the sum of the velocity mass (M) and stiffness (A) matrix;
 *       B^T is pressure gradient matrix
 *
 * As preconditioner we use the SPD matrix
 *
 * @f[
 * P = \left[ \begin{array}{cc} C & 0 \\ 0 & W \end{array} \right]
 * @f]
 *
 * where W is the diagonal of the pressure mass matrix.
 * The preconditioner P is applyied efficiently using one V-cycle of AMG (ml in Trilinos).
 *
 *
 * To run this code you must compile Trilinos. The required pakages are Epetra, EpetraExt, Teuchos, ML.
 *
 * USAGE:
 * mpirun -n 2 ./ex1.exe
 *
 */

int main(int argc,char * argv[])
{

	//(1) MPI initialization and allocation of the Epetra Communicator
	MPI_Init(&argc, &argv);
	std::cout<< "MPI Initialization\n";

	Epetra_MpiComm * comm(new Epetra_MpiComm(MPI_COMM_WORLD));

	// get process information
	int numProc = comm->NumProc();
	int myPID   = comm->MyPID();

	bool verbose( 0 == myPID );

	// Open a new scope: all Epetra Object must be deleted before the Epetra_Comm is deallocated
	// and the MPI session is closed
	{
		   // Define the map for the parallel vectors and operators.
		   int dim(2312);					//global dimension of the problem.
		   Epetra_Map map(dim, 1, *comm);	//define a linear map

		   //(1) Read the Stokes FE Matrix and Preconditioner from file.
		   Epetra_CrsMatrix * A = NULL;
		   Epetra_CrsMatrix * P = NULL;

		   if( verbose )
			   std::cout << "Reading matrix market file (Operator)" << std::endl;

		   int ierr(0);
		   ierr = EpetraExt::MatrixMarketFileToCrsMatrix("StokesMatrix.mtx",map,map,map,A);

		   if(ierr==0 && verbose)
			   std::cout<< "Read success \n";
		   if(ierr != 0 && verbose)
			   std::cout<<" Reading error " << ierr << "\n";

		   if( verbose )
			   std::cout << "Reading matrix market file (Preconditioner)" << std::endl;

		   ierr = EpetraExt::MatrixMarketFileToCrsMatrix("StokesPreconditioner.mtx",map,map,map,P);

		   if(ierr==0 && verbose)
			   std::cout<< "Read success \n";
		   if(ierr != 0 && verbose)
			   std::cout<<" Reading error " << ierr << "\n";

		   //(2) Create the MultiLevel preconditioner for the matrix P
		   Teuchos::ParameterList MLList;		//List of options for ml
		   ML_Epetra::SetDefaults("SA",MLList); //Use the Smoothed Aggregation defaults

		   ML_Epetra::MultiLevelPreconditioner * MLPrec(new ML_Epetra::MultiLevelPreconditioner(*P, MLList) ); //Create the actual preconditioner

		   //(3) Wrap the Epetra_Operator A and MLPrec with the adapters to be used in MINRES
		   EpetraOperatorAdapter op(*A);
		   EpetraPreconditionerAdapter * prec = new EpetraPreconditionerAdapter(*MLPrec);

		   //(4) Define the exact solution, the rhsm and the computed solution vector (and their wrapper)
		   Epetra_Vector ev_sol(map), ev_rhs(map), ev_x(map);
		   EpetraVectorAdapter sol(ev_sol), rhs(ev_rhs), x(ev_x);

		   sol.Randomize( 1 );	//Pick a random exact solution ||sol||_2 = 1

		   op.Apply(sol, rhs);	// rhs = op * sol
		   double rhsNorm2(0);
		   rhsNorm2 = InnerProduct(rhs, rhs);
		   if(verbose)
			   std::cout<<"|| rhs ||_2 = " << sqrt(rhsNorm2) << "\n";

		   //(5) Set up minres parameters
		   x = 0.0;				//Let the initial guess to be identically 0
		   double shift(0);		//No shift in minres
		   int max_iter(500);	//Set the maximum number of iterations
		   double tol(1.e-10);	//Set the tolerance of the stopping criterion

		   //(6) Solve the problem with minres
		   MINRES(op, x, rhs, prec, shift, max_iter, tol, verbose);

		   //(7) Compare against the exact solution
		   subtract(x, sol, x);
		   double err2 = InnerProduct(x,x);

		   if(verbose)
			   std::cout<< "|| x_ex - x_n || = " << sqrt(err2) << "\n";

		   delete prec;
		   delete MLPrec;
		   delete P;
		   delete A;


	}

	//(8) Dellacate the comm and finalize mpi
	delete comm;
	MPI_Finalize();

	return 0;
}
