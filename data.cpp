/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
 
#include <vector>
#include <algorithm>
#include "data.h"

using namespace std;

Transaction::Transaction(const Transaction &tr)
{
  length = tr.length;
  t = new int[tr.length];
  for(int i=0; i< length; i++)
    t[i] = tr.t[i];
}

Data::Data(char *filename)
{
  in = fopen(filename,"rt");
}

Data::~Data()
{
  if(in) fclose(in);
}

int Data::isOpen()
{
  if(in) return 1;
  else return 0;
}

Transaction *Data::getNext()
{
  vector<int> list;
  char c;

  // read list of items
  do {
    int item=0, pos=0;
    c = getc(in);
    while((c >= '0') && (c <= '9')) {
      item *=10;
      item += int(c)-int('0');
      c = getc(in);
      pos++;
    }
    if(pos) list.push_back(item);
  }while(c != '\n' && !feof(in));

  // if end of file is reached, rewind to beginning for next pass
  if(feof(in)){
    rewind(in);
    return 0;
  }
  // Note, also last transaction must end with newline,
  // else, it will be ignored

  // sort list of items (this is not necessary for the workshop test datasets)
  // sort(list.begin(),list.end());

  // put items in Transaction structure
  Transaction *t = new Transaction((int)list.size());
  for(int i=0; i<int(list.size()); i++)
    {
	  t->t[i] = list[i];
    }

  return t;
}
char* Data::getFileName() {
	return filename;
}
