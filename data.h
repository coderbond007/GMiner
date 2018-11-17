/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
class Transaction
{
public:
  Transaction(int l) : length(l) {t = new int[l];}
  Transaction(const Transaction &tr);
  ~Transaction(){delete [] t;}
  void printTransaction() {
	  for (int i =0; i < length; i++) {
		  cout << t[i]<< " ";
	  }
	  cout << endl;
  }

  struct TransactionCompare {
	  bool operator()(const Transaction* l, const Transaction* r) {
		  return l->length > r->length;
	  }
  };
  int length;
  int *t;
};

class Data
{
public:
  Data(char *filename);
  ~Data();
  int isOpen();
  Transaction *getNext();
  char* getFileName();
private:
  FILE *in;
  char* filename;
};
