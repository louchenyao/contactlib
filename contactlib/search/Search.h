#pragma once

#include "Library.h"
#include "Common.h"

#include <boost/dynamic_bitset.hpp>

struct Database
{
  int sizePROT;       // number of proteins
  vector<string> pid; // PDB ID of each protein
  vector<int> psize;  // number of contact groups of each protein

  vector<int> idxCONT2PROT;       // index converstion

  const int sizeSLOT = 161;                     // number of slots on each dimention
  int sizeCONT;                                 // number of contact groups in each slot
  vector<vector<boost::dynamic_bitset<> > > dbbs; // dimension_index, slot_index, contact_group_index

  Database()
  {
    sizePROT = 0;
    sizeCONT = 0;
  }
  void indexContact2Protein(const int &indexContact, int &indexProtein, int &pbegin, int &pend);
  void loadDB(const char *dbfn);
  void processDB(const int &cutoff);
  void search(const char *tfn, const char *res_fn);
};

extern "C" {
  void* newDB(const char *db_fn, int cutoff);
  void deleteDB(void *db);
  void search(void *db, const char *target_fn, const char *resutl_fn);
}
