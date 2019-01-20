#include "Library.h"
#include "Common.h"
#include "Search.h"

#include <boost/dynamic_bitset.hpp>

using namespace std;

const int sizeDIM = 7; // number of dimensions

typedef struct
{
  float dist;
  int index;
} Element;

void loadTarget(const char *tfn, vector<string> &tid, vector<vector<float> > &tdist, vector<vector<int> > &tindex)
{
  cerr << "loading target " << tfn << " ..." << endl;
  tid.clear();
  tdist.clear();
  tindex.clear();
  ifstream in(tfn);
  assert(in.is_open());
  string id;
  while (in >> id)
  {
    char tab;
    in.read((char *)&tab, 1);
    Element v[sizeDIM];
    in.read((char *)v, sizeof(Element) * sizeDIM);

    vector<float> bufferDist;
    vector<int> bufferIndex;
    for (int i = 0; i < sizeDIM; i++)
    {
      bufferDist.push_back(v[i].dist);
      bufferIndex.push_back(v[i].index);
    }

    tid.push_back(id);
    tdist.push_back(bufferDist);
    tindex.push_back(bufferIndex);
  }
  in.close();
  cerr << "--size: " << tid.size() << endl;
}

void Database::indexContact2Protein(const int &indexContact, int &indexProtein, int &pbegin, int &pend)
{
  // reset
  if (indexContact < pbegin)
  {
    indexProtein = 0;
    pbegin = 0;
    pend = psize[indexProtein];
  }

  // search
  while (indexContact >= pend)
  {
    indexProtein++;
    pbegin = pend;
    pend += psize[indexProtein];
  }
}

void Database::loadDB(const char *dbfn)
{
  cerr << "loading database " << dbfn << " ..." << endl;
  static const int MAX_BITSET_SIZE = 1024 * 1024;
  boost::dynamic_bitset<> v3(MAX_BITSET_SIZE);
  vector<boost::dynamic_bitset<>> v2;
  for (int i = 0; i <= sizeSLOT; i++)
    v2.push_back(v3);
  for (int i = 0; i < sizeDIM; i++)
    dbbs.push_back(v2);

  ifstream in(dbfn);
  assert(in.is_open());
  string id;
  sizePROT = 0;
  sizeCONT = 0;
  while (in >> id)
  {
    char tab;
    in.read((char *)&tab, 1);
    Element v[sizeDIM];
    in.read((char *)v, sizeof(Element) * sizeDIM);
    for (int i = 0; i < sizeDIM; i++)
    {
      while (sizeCONT >= dbbs[i][v[i].index].size())
      {
        dbbs[i][v[i].index].resize(dbbs[i][v[i].index].size() * 2);
      }
      dbbs[i][v[i].index].set(sizeCONT);
    }

    string pdb = id.substr(0, id.find(":"));
    idxCONT2PROT.push_back(sizePROT);
    if (!sizePROT || pdb.compare(pid[sizePROT - 1]))
    {
      pid.push_back(pdb);
      psize.push_back(1);
      sizePROT++;
    }
    else
    {
      psize[sizePROT - 1]++;
    }
    sizeCONT++;
  }
  in.close();

  for (int i = 0; i < sizeDIM; i++)
  {
    for (int j = 0; j < sizeSLOT; j++)
    {
      dbbs[i][j].resize(sizeCONT);
    }
  }
  cerr << "--size: " << sizeCONT << endl;
}

void Database::processDB(const int &cutoff)
{
  cerr << "processing database with cutoff " << cutoff << " ..." << endl;
  boost::dynamic_bitset<> bsdimension(sizeCONT);
  vector<boost::dynamic_bitset<>> bsslot;
  for (int i = 0; i < sizeSLOT; i++)
    bsslot.push_back(bsdimension);

  for (int i = 0; i < sizeDIM; i++)
  {
    for (int j = 0; j < sizeSLOT; j++)
    {
      bsslot[j].reset();

      int kbegin = max(0, j - cutoff);
      int kend = min(sizeSLOT, j + cutoff + 1);
      for (int k = kbegin; k < kend; k++)
      {
        bsslot[j] |= dbbs[i][k];
      }
    }
    for (int j = 0; j < sizeSLOT; j++)
    {
      dbbs[i][j] = bsslot[j];
    }
  }
}

struct OutputEntry {
  string a, b;
  double score;

  OutputEntry(string a, string b, double s):a(a), b(b), score(s) {}
  
  bool operator < (const OutputEntry & e) const {
    return score > e.score;
  }
};

void Database::search(const char *tfn, const char *res_fn)
{
  vector<string> tid;          // target contact group id
  vector<vector<float>> tdist; // target contact group distance vectors
  vector<vector<int>> tindex;  // target contact group distance vectors
  loadTarget(tfn, tid, tdist, tindex);
  vector<int> thits(sizePROT, 0); // pdbid, number of target hits
  vector<int> dbhits(sizePROT, 0); // pdbid, number of database hits
  vector<int> hits(sizePROT, 0);  // pdbid, number of hits between target and database
  boost::dynamic_bitset<> bsstruct(sizeCONT);
  boost::dynamic_bitset<> bscontact(sizeCONT);
  // int indexProtein = 0;
  // int pbegin = 0; // for indexContac2Proteion;
  // int pend = psize[0];

  // find alignments, calculate number of target/database hits
  cerr << "looking for alignments ..." << endl;
  bsstruct.reset();
  #pragma omp parallel for
  for (int i = 0; i < tindex.size(); i++)
  {
    bscontact.set();
    for (int j = 0; j < sizeDIM; j++)
    {
      bscontact &= dbbs[j][tindex[i][j]];
    }
    #pragma omp critical (bsstruct)
    bsstruct |= bscontact;

    int indexPrev = -1;
    for (int j = bscontact.find_first(); j != bscontact.npos; j = bscontact.find_next(j))
    {
      //indexContact2Protein(j, indexProtein, pbegin, pend);
      int indexProtein = idxCONT2PROT[j];
      //cerr << indexProtein << endl;
      #pragma omp atomic update
      hits[indexProtein]++;
      if (indexProtein == indexPrev)
        continue;
      #pragma omp atomic update
      thits[indexProtein]++;
      indexPrev = indexProtein;
    }
  }
  for (int i = bsstruct.find_first(); i != bsstruct.npos; i = bsstruct.find_next(i))
  {
    //indexContact2Protein(i, indexProtein, pbegin, pend);
    int indexProtein = idxCONT2PROT[i];
    dbhits[indexProtein]++;
  }

  // output
  cerr << "printing alignments ..." << endl;
  fstream res;
  res.open(res_fn, fstream::out);
  vector<OutputEntry> oe;
  for (int i = 0; i < sizePROT; i++)
  {
    oe.push_back(OutputEntry(tid[0].substr(0, tid[0].find(":")), pid[i], sqrt(1.0 * thits[i] * dbhits[i] / tid.size() / psize[i]))); 
  }
  sort(oe.begin(), oe.end());
  for (int i = 0; i < oe.size(); i++) {
    res << oe[i].a << "\t" << oe[i].b << "\t" << oe[i].score << endl;
  }
  res.close();
}

// extern "C" {

void search(void *db, const char *target_fn, const char *resutl_fn)
{
  cerr << __LINE__ << " " << db << endl;
  ((Database *)(db))->search(target_fn, resutl_fn);
}

// We want to export C style ABI, Database isn't.
// So return void* instead of Database*.
void* newDB(const char *db_fn, int cutoff)
{
  Database *db = new Database();
  db->loadDB(db_fn);
  db->processDB(cutoff);
  return (void*)db;
}

void deleteDB(void *db)
{
  delete ((Database *)(db));
}

// } // extern "C"