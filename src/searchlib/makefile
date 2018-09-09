CPP=g++ -O3 --std=c++11

all: search

search: main.cpp libsearch.so Search.h Common.h
	$(CPP) -o $@ main.cpp -L. -lsearch

libsearch.so: Search.o Common.o
	$(CPP) -o libsearch.so -shared Search.o Common.o

Search.o: Search.cpp Search.h Common.h
	$(CPP) -o $@ -c Search.cpp

Common.o: Common.cpp Common.h
	$(CPP) -o $@ -c Common.cpp

clean:
	rm -rf *.o search *.so
