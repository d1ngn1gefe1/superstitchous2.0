# Created on: Apr 16, 2012
#     Author: uvilla

export RELEASE_DIR=tminres-0.2
export MAKE=make

$MAKE dox;
mkdir $RELEASE_DIR;
cp *.hpp $RELEASE_DIR;
cp README $RELEASE_DIR;
cp Doxyfile.in  $RELEASE_DIR;
cp Mainpage.dox $RELEASE_DIR;
cp Makefile $RELEASE_DIR;
mkdir $RELEASE_DIR/SerialExample;
cp SerialExample/*.hpp $RELEASE_DIR/SerialExample;
cp SerialExample/*.cpp $RELEASE_DIR/SerialExample;
cp SerialExample/Makefile $RELEASE_DIR/SerialExample;
mkdir $RELEASE_DIR/TrilinosExample;
cp TrilinosExample/*.hpp $RELEASE_DIR/TrilinosExample;
cp TrilinosExample/*.cpp $RELEASE_DIR/TrilinosExample;
cp TrilinosExample/*.mtx $RELEASE_DIR/TrilinosExample;
cp TrilinosExample/Makefile $RELEASE_DIR/TrilinosExample;
cp TrilinosExample/Makefile.in $RELEASE_DIR/TrilinosExample;
cp -r doc $RELEASE_DIR/doc;
rm -rf $RELEASE_DIR/doc/html/formula.repository;

cd $RELEASE_DIR
$MAKE serial
$MAKE trilinos
cd SerialExample;
./ex1.exe;
./ex2.exe;
$MAKE clean;
cd ../TrilinosExample;
mpirun -n 4 ./ex1.exe;
$MAKE clean;
