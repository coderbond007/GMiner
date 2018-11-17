/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#pragma once

#include <vector>
#include "Global.h"
using namespace std;

// representing itemset gotton from the implementation of Goethals
class Item
{
 public:

  Item(int i, unsigned s = 0) : id(i), support(s) {}

  inline bool operator< (const Item &i) const {return support < i.support;}

  mutable vector<unsigned> transactions;
  int id;
  unsigned support;
};
