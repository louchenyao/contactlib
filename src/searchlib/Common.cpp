#ifndef COMMON
#define COMMON

#include "Library.h"
#include "Common.h"

using namespace std;


const int ADDRESS_SIZE = sizeof(void*);

// malloc 2d array
void** malloc2(const int& typesize, const int& len1, const int& len2, const bool& reset = false) {
  int arraysize2 = typesize * len2;
  int arraysize1 = (arraysize2 + ADDRESS_SIZE) * len1;

  void** result = (void**) malloc(arraysize1);
  unsigned long addr = (unsigned long) result;
  addr += ADDRESS_SIZE * len1;
  for (int i = 0; i < len1; i++) {
    result[i] = (void*) addr;
    addr += arraysize2;
  }
  if (reset) memset(result[0], 0, arraysize2 * len1);
  return result;
}

// malloc 3d array
void*** malloc3(const int& typesize, const int& len1, const int& len2, const int& len3, const bool& reset = false) {
  int arraysize3 = typesize * len3;
  int arraysize2 = (arraysize3 + ADDRESS_SIZE) * len2;
  int arraysize1 = (arraysize2 + ADDRESS_SIZE) * len1;

  void*** result = (void***) malloc(arraysize1);
  unsigned long addr = (unsigned long) result;
  addr += ADDRESS_SIZE * len1;
  for (int i = 0; i < len1; i++) {
    result[i] = (void**) addr;
    addr += ADDRESS_SIZE * len2;
  }
  for (int i = 0; i < len1; i++) {
    for (int j = 0; j < len2; j++) {
      result[i][j] = (void*) addr;
      addr += arraysize3;
    }
  }
  if (reset) memset(result[0][0], 0, arraysize3 * len2 * len1);
  return result;
}

// convert radian to degree
const float RAD2DEG(const float& rad) {
  return rad / M_PI * 180.0;
}

// convert degree to radian
const float DEG2RAD(const float& deg) {
  return deg / 180.0 * M_PI;
}

// angle difference
const float diffAngle(const float& a, const float& b) {
  float diff = fabs(fmod(a - b, float(2.0 * M_PI)));
  return (diff < M_PI) ? diff : 2.0 * M_PI - diff;
}

// convert distance to angle
const float dist2angle(const float& dist, const float& radius) {
  return asin(dist / 2.0 / radius) * 2.0;
}

// round float to integer
const int float2int(const float& value) {
  return int(floor(value + 0.5));
}

// load filename from a list file
vector<string> loadFilenames(const string& fn) {
  vector<string> result;
  if (fn.substr(fn.size() - 3).compare("lst")) {
    result.push_back(fn);
  } else {
    ifstream in(fn.c_str());
    assert(in.is_open());
    string buffer;
    while (in >> buffer) result.push_back(buffer);
    in.close();
  }
  return result;
}

// show current progress
void progress(const int& curr, const int& size) {
  int prev = 100 * curr / size;
  int next = 100 * (curr + 1) / size;
  if (prev == next) return;
  cerr << ".";
  if (next % 20 == 0) cerr << "  " << next << "%" << endl;
}

// random generator of Gaussian(0, 1)
const float gaussian01() {
  float x1, x2, w, y1;
  static float y2;
  static bool ready = false;

  if (ready) {
    ready = false;
    return y2;
  } else {
    do {
      x1 = 2.0 * rand() / RAND_MAX - 1.0;
      x2 = 2.0 * rand() / RAND_MAX - 1.0;
      w = x1 * x1 + x2 * x2;
    } while (w >= 1.0);
    w = sqrt((-2.0 * log(w)) / w);
    y1 = x1 * w;
    y2 = x2 * w;
    ready = true;
    return y1;
  }
}

#endif // COMMON

