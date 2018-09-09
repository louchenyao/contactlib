CPP=g++ -static -O3

all: search

search: Search.cpp Common.o
	$(CPP) -o $@ $?

Common.o: Common.cpp Common.h
	$(CPP) -o $@ -c Common.cpp

clean:
	rm -rf *.o search
