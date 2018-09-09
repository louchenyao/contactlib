#include "Library.h"
#include "Common.h"
#include "Search.h"

using namespace std;

int main(int argc, char **args)
{
  // parse parameters
  assert(argc == 3);
  int index = 1;
  string tfn = args[index++];
  string dbfn = args[index++];
  int cutoff = atoi(args[index++]);

  // read target and database
  cerr << "#preprocessing" << endl;
  void *db = newDB(dbfn.c_str(), cutoff);

  vector<string> tfns = loadFilenames(tfn);
  for (int tfnIndex = 0; tfnIndex < tfns.size(); tfnIndex++)
  {
    cerr << "#working on target " << tfnIndex << "/" << tfns.size() << endl;
    ((Database*)(db))->search(tfns[tfnIndex]);
  }
  cerr << "done!!!" << endl;
}
