#ifndef LIBRARY
#define LIBRARY

// macros
#define POW2(n) (n * n)
#define POW3(n) (n * n * n)
#define DEBUG(n) (cerr << "DEBUG\t<" << #n << ">\t<" << n << ">" << endl)

// libraries
#include <string>

#include <set>
#include <map>
#include <vector>
#include <bitset>

#include <sstream>
#include <fstream>
#include <iostream>

#include <cmath>
#include <cfloat>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cassert>

// namespace
using namespace std;

#ifdef MEMORY_DEBUG
static int libCount = 0;

void* operator new(size_t size) {
  libCount++;
  void* ptr = malloc(size);
  return ptr;
}

void operator delete(void* ptr) {
  libCount--;
  free(ptr);
}
#endif // MEMORY_DEBUG

#endif // LIBRARY

