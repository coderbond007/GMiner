/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#ifndef VERTICALDATABASE_H_
#define VERTICALDATABASE_H_
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <map>

 #include "data.h"
#include "item.h"
#include "Global.h"
using namespace std;

class verticalDatabase {
public:
	verticalDatabase() : data(0), out(0), numOfSingletons(0) {
	}
	~verticalDatabase() {
		if (data) {
			delete data;
		}
		if (out) {
			fclose(out);
		}
	}

	void setData(char *file) {
		data = new Data(file);

	}

	void setMinsupRelative(float ms) {
		minSupRelative = ms;
	}
	void setOutput(char* of) {
		out = fopen(of, "wt");
	}
	void buildVerticalDatabase(int _fragmentWidth) {
 		unsigned tnr = 0;
		set<Item> root;
 		set<Item>::iterator it;

		//	Timer startSortDB, endSortDB;
		Timer startLoadDB = getCurrentTime();
		while (Transaction *t = data->getNext()) {
			for (int i = 0; i < t->length; i++) {
				it = root.find(Item(t->t[i], t->t[i]));
				if (it == root.end()) {
					it = root.insert(Item(t->t[i], t->t[i])).first;
				}
				it->transactions.push_back(tnr);
			}
			tnr++;
			delete t;
		}
		cout <<"Reading file into main memory. [";
		Timer endLoadDB = getCurrentTime();
		cout <<(endLoadDB - startLoadDB)<<" sec]" << endl;
		//cout << "The number of transactions is " << tnr << endl;

		minSupAbsolute = minSupRelative * tnr;

		// remove infrequent items and put items in support ascending order
		Timer startFrequentSingletons = getCurrentTime();
		while ((it = root.begin()) != root.end()) {
			if (it->transactions.size() >= minSupAbsolute) {
				numOfSingletons++;
				Item item(it->id, it->transactions.size());
				item.transactions = it->transactions;
				allitems.push_back(item);
			}
			root.erase(it);
		}
		sort(allitems.begin(), allitems.end());
		Timer endFrequentSingletons = getCurrentTime();
		cout <<"Finding frequent 1-itemsets. [" << (endFrequentSingletons - startFrequentSingletons) << " sec]" << endl;

		int remain = allitems.size() % _fragmentWidth;
		int padding_num = _fragmentWidth - remain;
		for (int idx = 0; idx <padding_num; idx++) {
			Item item(-1,0);
			allitems.insert(allitems.begin(), item);
		}
		int remap_item = 0;
		for (int remapIdx = 0; remapIdx < allitems.size() ; remapIdx++) {
			mapFrequentSingletons[remap_item] = allitems[remapIdx].id;
			allitems[remapIdx].id = remap_item;
			singleton.push_back(remap_item);
			mapSingletonIndex[remapIdx] = remapIdx;
			remap_item++;
			probabilityOfSingletons.push_back( (double) allitems[remapIdx].support / (double) tnr);
		}

		if (out) {
			fprintf(out, "(%d)\n", tnr);
		}
		tidCount = tnr;
	}

 	vector<Item> allitems;
	vector<uint> singleton;
 	Data *data;
	FILE* out;

	// the number of transactions in the transaction database
	unsigned tidCount;

	// relative minimum support specified by the user
	float minSupRelative;

	// absolute minimum support
	unsigned minSupAbsolute;

	// the number of frequent items
	unsigned numOfSingletons;
	map<int, int> mapFrequentSingletons;
	map<uint, uint> mapSingletonIndex;
	vector<double> probabilityOfSingletons;
private:
};

#endif /* VERTICALDATABASE_H_ */
