CC = nvcc
C_FLAGS = -g -arch=sm_20 
L_FLAGS = -lcuda -lcusparse
I_DIR = -I ../ -I . 

all : testVector.exe cusparse_example1.exe cusparse_example2.exe cusparse_example3.exe 

testVector.exe : CusparseVector.o testVector.o CusparseOperator.o CusparseSymmetricOperator.o
	$(CC) $^ -o $@ $(C_FLAGS) $(L_FLAGS)

cusparse_example1.exe: CusparseVector.o CusparseOperator.o cusparse_example1.o
	$(CC) $^ -o $@ -arch=sm_20 -lcusparse -lcuda

cusparse_example2.exe: CusparseVector.o CusparseOperator.o CusparseSymmetricOperator.o cusparse_example2.o
	$(CC) $^ -o $@ -arch=sm_20 -lcusparse -lcuda

cusparse_example3.exe: CusparseVector.o mmio.o CusparseOperator.o cusparse_example3.o 
	$(CC) $^ -o $@ -arch=sm_20 -lcusparse -lcuda

%.o : %.cu
	$(CC) -c $(C_FLAGS) $(I_DIR) $^ 

clean :
	rm *.o *.exe
