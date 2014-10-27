CC = nvcc
C_FLAGS = -g -arch=sm_20
I_DIR = -I ../ 

all : testCuda.exe cuda_example1.exe 

testCuda.exe : testCuda.o CudaVector.o CudaOperator.o
	$(CC) $^ -o $@ $(I_DIR) -lcuda -lcublas

cuda_example1.exe : cuda_example1.o CudaVector.o CudaOperator.o 
	$(CC) $^ -o $@ $(I_DIR) -lcuda -lcublas

%.o : %.cu
	$(CC) -c $(C_FLAGS) $(I_DIR) $^ 


clean :
	rm *.o *.exe 
