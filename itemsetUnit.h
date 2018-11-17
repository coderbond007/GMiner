/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#ifndef ITEMSETUNIT_H_
#define ITEMSETUNIT_H_

#include <vector>
#include "Global.h"
using namespace std;

class itemsetUnit {
public:
	itemsetUnit(){}
	itemsetUnit(vector<uint> _itemset) {
		itemset = _itemset;
		isFrequent = false;
	}
	inline void setStatusOfFrequent(bool _isFrequent) {
		isFrequent = _isFrequent;
	}
	inline void setItemset(vector<uint> _itemset) {
		itemset = _itemset;
	}
	inline vector<uint> getItemset(){
		return itemset;
	}
	inline bool getStatusOfFrequent() {
		return isFrequent;
	}
	inline bool operator==(const itemsetUnit& other) const {
		if(this->itemset == other.itemset) {
			return true;
		}
		else {
			return false;
		}
	}
	virtual ~itemsetUnit();
public:
	bool isFrequent;
	vector<uint> itemset;
private:


};

#endif /* ITEMSETUNIT_H_ */
