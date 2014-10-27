serial :
	cd SerialExample; $(MAKE) all;
	
trilinos :
	cd TrilinosExample; $(MAKE) all;

dox :
	doxygen Doxyfile.in;
	
clean :
	cd SerialExample; $(MAKE) clean;
	cd TrilinosExample; $(MAKE) clean;
	
clean-dox :
	rm -r doc;