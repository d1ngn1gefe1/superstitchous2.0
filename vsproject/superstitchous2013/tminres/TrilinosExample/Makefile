include Makefile.in

all : ex1.exe testEpetraVectorAdapter.exe

ex1.exe : ex1.o EpetraVectorAdapter.o
	$(MPICC) $(C_FLAG) $^ -o $@ $(LIBS)

testEpetraVectorAdapter.exe : testEpetraVectorAdapter.o EpetraVectorAdapter.o
	$(MPICC) $(C_FLAG) $^ -o $@ $(LIBS)
	
%.o : %.cpp
	$(MPICC) -c $(C_FLAG) $(INCLUDE_FLAG) $^

clean : 
	rm -f *.exe *.o
