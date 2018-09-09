#pragma once


extern "C" {
  int init(char *db_name, int cutoff);
  int find_match(char *target_dnn_path, int dim, char *name);
}
