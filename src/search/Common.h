#ifndef COMMON
#define COMMON

#include "Library.h"

const int ADDRESS_SIZE = sizeof(void*);

// malloc 2d array
void** malloc2(const int& typesize, const int& len1, const int& len2, const bool& reset = false);

// malloc 3d array
void*** malloc3(const int& typesize, const int& len1, const int& len2, const int& len3, const bool& reset = false);

// convert radian to degree
const float RAD2DEG(const float& rad);

// convert degree to radian
const float DEG2RAD(const float& deg);

// angle difference
const float diffAngle(const float& a, const float& b);

// convert distance to angle
const float dist2angle(const float& dist, const float& radius);

// round float to integer
const int float2int(const float& value);

// load filename from a list file
vector<string> loadFilenames(const string& fn);

// show current progress
void progress(const int& curr, const int& size);

// random generator of Gaussian(0, 1)
const float gaussian01();

#endif // COMMON

