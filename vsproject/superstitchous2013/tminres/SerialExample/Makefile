CC = g++
C_FLAGS = -g
I_DIR = -I ../

all : ex1.exe ex2.exe ex3.exe testSimpleVector.exe 

ex1.exe :ex1.o SimpleVector.o
	$(CC) $^ -o $@

ex2.exe : ex2.o SimpleVector.o
	$(CC) $^ -o $@ 

ex3.exe : ex3.o SimpleVector.o SimpleOperator.o
	$(CC) $^ -o $@ 

testSimpleVector.exe : testSimpleVector.o SimpleVector.o
	$(CC) $^ -o $@

%.o : %.cpp
	$(CC) -c $(C_FLAGS) $(I_DIR) $^ 

clean :
	rm -f *.o *.exe *.m
