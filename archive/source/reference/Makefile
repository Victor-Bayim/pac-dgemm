PWD = $(shell pwd)
all:lib dgemmtest

lib:
	g++ src/gemm.cpp -shared -o libgemm.so -fPIC -I ./include/ -fopenmp

dgemmtest: lib
	g++ test/gemmtest.cpp test/help.cpp -I ./include/ -I ./test/ $(PWD)/libgemm.so -o dgemmbench -DFLOAT64

clean:
	rm -rf *.so *.o test/*.o dgemmbench