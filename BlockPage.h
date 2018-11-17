/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#ifndef BLOCKPAGE_H_
#define BLOCKPAGE_H_

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

struct ElementInfo {
public:
	ElementInfo(){}
	ElementInfo(vector<uint> _itemset, bool _status, uint _block_position, uint _ROW_ID) {
		itemset = _itemset;
		status = _status;
		block_position = _block_position;
		ROW_ID = _ROW_ID;
	}
	void updateStatus(bool _status) {
		status = _status;
	}
	void printElementInfo() {
		cout <<"itemset - ";
		for(int idx = 0; idx < itemset.size();idx++) {
			cout << itemset[idx]<<" ";
		}
		cout <<", status - " << status<<", block_position - "<< block_position << ", ROW_ID - " << ROW_ID<<endl;
	}

public:
	vector<uint> itemset;  // itemset
	bool status;   // status whether itemset is frequent or not
	uint block_position; // position in a BlockPage
						 // note that block_positions of the itemset in all BlockPages are the same as each other
	uint ROW_ID;
	//uint global_position; // position in global view; used in each level
};
class BlockPage {
public:
	BlockPage()	{
 	}

	void configureBlockPage(vector<uint> _singletonVec, uint _ROW_ID, uint _PAGE_ID, uint _WIDTH, uint _HEIGHT, bool _isMaterialization) {
		isMaterialization = _isMaterialization;
		singletonVec = _singletonVec;
		ROW_ID = _ROW_ID;
		PAGE_ID = _PAGE_ID;
		verticalListLength = _WIDTH;
		rowLength = _HEIGHT;
		MAX_ITEM = singletonVec[singletonVec.size()-1];
		MIN_ITEM = singletonVec[0];
		if (isMaterialization == true) {
			numOfItemsets = pow((double)2, (double)_singletonVec.size())-1;
			// build the dictionary each of which element includes <I, info(I)>
			// info(I) includes itemset, position in each block, and status of I with respect to whether it is frequent or not

			sort(singletonVec.rbegin(), singletonVec.rend());
			setEnumeration(singletonVec);
		}
		else {
			numOfItemsets = singletonVec.size();
		}
 

		if (MAX_ITEM < MIN_ITEM) {
			cerr <<"configureBlockPage():: MAX_ITEM < MIN_ITEM! Rewrite codes" << endl;
			cerr <<"configureBlockPage():: MIN_ITEM - " << MIN_ITEM <<", MAX_ITEM - " << MAX_ITEM << endl;
		}


	}
	void MallocBlockPage() {
		cudaError_t cudaStatus;
		cudaStatus = cudaHostAlloc((void**)&dataForPage, sizeof(uint)*verticalListLength*rowLength, cudaHostAllocDefault);
		if (cudaStatus != cudaSuccess) {
			cerr <<"MallocBlockPage():: Error in cudaHostAlloc for dataForPage"<<endl;
		}
		memset(dataForPage, 0, sizeof(uint)*verticalListLength*rowLength);
	}
 
	void deMallocBlockPage() {
		cudaError_t cudaStatus;
		cudaStatus = cudaFreeHost(dataForPage);
		if(cudaStatus!=cudaSuccess) {
			cerr <<"deMallocBlockPage():: Error in cudaFreeHost for dataForPage" << endl;
		}
		dataForPage = NULL;
	}


	~BlockPage() {
		//cout <<"~BlockPage():: deconstructing a BlockPage" << endl;
	}

	void setEnumeration(vector<uint> arr) {
		int positionInBlockCnt = 0;
		int len = arr.size();
 		int flag = 1;
		unsigned int max = 1 << len;
		for(;flag < max; flag++) {
			vector<uint> v;
			int v_idx=0;
			int idx = 0;
			int t = flag;
			while (t) {
				++idx;
				if(t&0x1) {
 					v.push_back(arr[len-idx]);
				}
				t = t >> 1;
			}
			ElementInfo ei(v, true, positionInBlockCnt, ROW_ID);
			ElementInfoMap[v] = ei;
			positionInBlockCnt++;
		}
		//cout <<"setEnumeration():: the number of singletons - " << len << ", the number of itemsets pre-computed - " << positionInBlockCnt<<endl;
	}

	void printBlockPage()
	{
		cout <<"MAX_ITEM - " << MAX_ITEM <<", MIN_ITEM - " << MIN_ITEM <<", ROW_ID - " << ROW_ID <<", PAGE_ID - " << PAGE_ID <<", WIDTH - " << verticalListLength <<", HEIGHT - " << rowLength << ", numofitemsets - " << numOfItemsets<< endl;
	}
public:
	vector<uint> singletonVec;
	uint* dataForPage;
 	map<vector<uint>, ElementInfo> ElementInfoMap;
	// MAX_ITEM and MIN_ITEM are used to filter unnecessary BLOCK ROWs
	uint MAX_ITEM;
	uint MIN_ITEM;

	uint ROW_ID;
	uint PAGE_ID;
	uint verticalListLength;  // vertical list length
	uint rowLength; // the fixed length of BLOCK ROWs
	uint numOfItemsets ; // the number of itemsets pre-computed
	bool isMaterialization;
private:

};


class BlockPageManager {
public:
	BlockPageManager()
	{
	}
 
	void configureBlockPageManager(vector<uint> _allSingletonVec, uint _verticalListLength, uint _PAGE_NUM, uint _FRAGMENT_SIZ, bool _isMaterialization) {
		cout <<"configureBlockPageManager():: check input parameters" << endl;
		cout <<"_allSingletonVec.size() - " << _allSingletonVec.size() <<", _verticalListLength - " << _verticalListLength <<", _PAGE_NUM - " << _PAGE_NUM <<", _FRAGMENT_SIZ - " <<_FRAGMENT_SIZ << endl;

		allSingletonVec = _allSingletonVec;
		verticalListLength = _verticalListLength;
		isMaterialization = _isMaterialization;
	
		// computing the number of itemsets pre-computed in each fragment
		if(isMaterialization == true) {
			rowLength = pow((double)2, (double)_FRAGMENT_SIZ) - 1;
		}
		else {
			rowLength = _FRAGMENT_SIZ;
		}

		PAGE_NUM = _PAGE_NUM;
		FRAGMENT_SIZ = _FRAGMENT_SIZ;
		ROW_NUM = ((allSingletonVec.size()-1)/_FRAGMENT_SIZ + 1);
		BLK_SIZ = verticalListLength * rowLength * 4;
		cout <<"configureBlockPageManager():: check variables"<<endl;
		cout <<"verticalListLength - " << verticalListLength <<", numOfItemsets - "				<< rowLength <<", PAGE_NUM - " << PAGE_NUM				<<", FRAGMENT_SIZ - " << FRAGMENT_SIZ <<", ROW_NUM - " <<ROW_NUM<<", BLK_SIZ - " << BLK_SIZ<<endl;

		//cout <<"configureBlockPageManager():: Complete" << endl;
	}
	void initBlockPageManager()	{

		// reservingthe memory for BlockPageVec
		int loopLimit = PAGE_NUM * ROW_NUM;
		for(int loopIdx = 0; loopIdx < loopLimit; loopIdx++) {
			BlockPage* bp_tmp = new BlockPage();
			BlockPageVec.push_back(bp_tmp);
		}

		int rowCnt = 0;

		for (int fragmentIdx = 0 ; fragmentIdx < this->ROW_NUM; fragmentIdx++) {
			vector<uint> fragmentVector;
			for (int idx = fragmentIdx * FRAGMENT_SIZ; idx < (fragmentIdx+1) * FRAGMENT_SIZ; idx++) {
				if (allSingletonVec.size()-1 < idx ){
					break;
				}
				fragmentVector.push_back(allSingletonVec[idx]);
				mapSingletonToBlkID[allSingletonVec[idx]] = rowCnt;
			}
			uint num_itemset = 0;
			if (isMaterialization == true) {
				num_itemset = pow((double)2, (double)fragmentVector.size()) -1 ;
			}
			else {
				num_itemset = fragmentVector.size();
			}

			this->initPartialBlockPageInManager(fragmentVector, rowCnt, verticalListLength, rowLength, num_itemset);

			rowCnt++;
		}
	}

	void initPartialBlockPageInManager (vector<uint> _blockSingletonVec, uint _ROWIDX, uint _WIDTH, uint _HEIGHT, uint _NUM_ITEM ) {
		for (int pageIdx = 0; pageIdx < PAGE_NUM; pageIdx++) {
			BlockPageVec[_ROWIDX * PAGE_NUM + pageIdx]->configureBlockPage(_blockSingletonVec, _ROWIDX, pageIdx, _WIDTH, _HEIGHT, isMaterialization);
			BlockPageVec[_ROWIDX * PAGE_NUM + pageIdx]->MallocBlockPage();
		}
	}
	void destroyBlockPageManager()
	{
		int loopLimit = PAGE_NUM * ROW_NUM;
		for(int loopIdx = 0; loopIdx < loopLimit; loopIdx++) {
			BlockPageVec[loopIdx]->deMallocBlockPage();
			delete BlockPageVec[loopIdx];
			BlockPageVec[loopIdx] = NULL;
		}
		vector<BlockPage*> s;
		s.swap(BlockPageVec);

	}

	~BlockPageManager()
	{
		//cout <<"~BlockPageManager()" << endl;
	}

public:
	vector<BlockPage*> BlockPageVec;
	vector<uint> allSingletonVec;
	uint verticalListLength;  // vertical list length
	uint rowLength; 		  // the number of itemsets to be pre-computed
	uint PAGE_NUM;			  // the number of pages
	uint ROW_NUM; 			  // the number of ROWs
	uint BLK_SIZ; 			  // the size of each BLockPage
	uint FRAGMENT_SIZ; 		  // the number of itemsets in each fragment
	bool isMaterialization;
	map<uint, uint> mapSingletonToBlkID;
private:

};

class rowBlock {
public:
	rowBlock() {}
	~rowBlock() {}
	void initRowBlock(bool _isMaterialization, uint _rowID, uint _rowSize, vector<uint> _blkSingletonVec) {
		indexCnt = 0;
		isMaterialization = _isMaterialization;
		isUsedInPreviousIteration = true;
		rowID = _rowID;
		rowIDActive = 0;
		rowSize = _rowSize;
		if (isMaterialization == true) {
			rowBlockSize = sizeof(uint) * (pow((double)2, (double)_blkSingletonVec.size()) - 1);
			setEnumeration(_blkSingletonVec);
		}
		else {
			rowBlockSize = sizeof(uint) * _blkSingletonVec.size();
			for (int idx = 0; idx < _blkSingletonVec.size(); idx ++ ) {
				vector<uint> tmpVec;
				tmpVec.push_back(_blkSingletonVec[idx]);
				mapItemsetToPositionInBLK[tmpVec] = indexCnt++;
			}
		}
	}
	void updateRowIDActive(uint _rowIDActive) {
		rowIDActive = _rowIDActive;
	}
public:
	bool isMaterialization;
	bool isUsedInPreviousIteration;
	uint rowID;
	// rowIDActive is used to adjust the positions in the global view
	int rowIDActive;
	uint rowSize;
	uint rowBlockSize;
	map<vector<uint>, uint> mapItemsetToPositionInBLK;
	uint indexCnt;
private:
	void setEnumeration(vector<uint> arr) {
		if(arr.size()==1){
 			mapItemsetToPositionInBLK[arr] = indexCnt;
 			indexCnt++;
 			return;
		}
		int len = arr.size();
		//cout <<"setEnumeration():: len - " << len<<endl;
		int flag = 1;
		unsigned int max = 1 << len;
		for(;flag < max; flag++) {
			vector<uint> v;
			int v_idx=0;
			int idx = 0;
			int t = flag;
			while (t) {
				++idx;
				if(t&0x1) {
					v.push_back(arr[len-idx]);
				}
				t = t >> 1;
			}
			mapItemsetToPositionInBLK[v] = indexCnt;
			indexCnt++;
		}

	}

};

class rowBlockManager {
public:
	rowBlockManager()
	{

	}
	~rowBlockManager()
	{

	}
	void initRowBlockManager (bool _isMaterialization, vector<Item> _singletons, uint _widthOfRows) {
		isMaterialization = _isMaterialization;
		widthOfRows = _widthOfRows ;
		singletons = _singletons;
		numOfRows = (singletons.size() - 1) / widthOfRows + 1;
		//computing the number of itemset pre-computed
		if(isMaterialization == true) {
			numOfItemsetsMaterialized = pow((double)2, (double)widthOfRows) - 1;
		}
		else {
			numOfItemsetsMaterialized = widthOfRows;
		}

		vector<uint> fragmentSize;
		// recording the criteria 
		for (int idx = 0; idx < numOfRows; idx++ ) {
			fragmentSize.push_back(widthOfRows);
		}
		uint boundaryCnt = 0;
		for(int idx = 0; idx < fragmentSize.size(); idx++) {
			boundaryCnt += fragmentSize[idx];
			boundary.push_back(boundaryCnt);
		}

		vector<uint> v;
		uint boundaryIdx =0;
		uint rowIDX = 0;
		
		// assigning rowIdx for each frequent 1-itemsets for discarding unnecessary blocks later	
		for(int idx = 0; idx < _singletons.size(); idx++) {
			if (_singletons[idx].id < boundary[boundaryIdx]) {
				v.push_back(_singletons[idx].id);
				mapSingletonToBlkID[_singletons[idx].id] = rowIDX;
				if(idx == _singletons.size()-1)
				{
					sort(v.rbegin(), v.rend());
					rowBlock rb;
					rb.initRowBlock(isMaterialization,rowIDX++,widthOfRows,v);
					rowBlockVector.push_back(rb);
				}
			}
			else {
				sort(v.rbegin(), v.rend());
				rowBlock rb;
				rb.initRowBlock(isMaterialization,rowIDX++,widthOfRows,v);
				rowBlockVector.push_back(rb);
				v.clear();
				boundaryIdx++;
				idx--;
			}
		}

		for (int idx = 0; idx < rowBlockVector.size(); idx++) {
			map<vector<uint>, uint> ::iterator row_iter;
			map<vector<uint>, uint> m  = rowBlockVector[idx].mapItemsetToPositionInBLK;
			row_iter = m.begin();
			while(row_iter != m.end()) {
				mapItemsetToPositionInAll[row_iter->first] = row_iter->second + (idx * numOfItemsetsMaterialized);
				row_iter++;
			}
		}
	}

	
	void setActiveRowBlock(set<uint> singletonUsedInIteration) {
		active_blk.clear();
		set<uint>::iterator it = singletonUsedInIteration.begin();
		map<uint, uint>::iterator mapSingletonTOBlkIDIterator ;
		while(it != singletonUsedInIteration.end()) {
			mapSingletonTOBlkIDIterator = mapSingletonToBlkID.find(*it);
			// finding active blocks for the current iteration
			if (mapSingletonTOBlkIDIterator != mapSingletonToBlkID.end()) {
				active_blk.insert(mapSingletonTOBlkIDIterator->second);
			}
			it++;
		}
		it = active_blk.begin();
		uint rowIDActiveCnt = 0;
		// modifying rowIDs each of which is used the current iteration
		while(it != active_blk.end()) {
			rowBlockVector[*it].rowIDActive = rowIDActiveCnt++;
			it++;
		}

	}

	// reorganizing position sets for not considering unactive blocks
	void reorganize_positionInAll() {
		mapItemsetToPositionInAll.clear();
		set<uint>::iterator iter = active_blk.begin();
		while(iter != active_blk.end()) {
			rowBlock rb = rowBlockVector[*iter];
			map<vector<uint>, uint>::iterator blk_iterator = rb.mapItemsetToPositionInBLK.begin();
			while(blk_iterator != rb.mapItemsetToPositionInBLK.end()) {
				mapItemsetToPositionInAll[blk_iterator->first] = blk_iterator->second + (rb.rowIDActive * numOfItemsetsMaterialized);
				blk_iterator++;
			}
			iter++;
		}

	}


public:
	bool isMaterialization;
	uint numOfRows;
	uint widthOfRows;
	uint numOfItemsetsMaterialized;
	vector<Item> singletons;
	vector<rowBlock> rowBlockVector;
	map<vector<uint>, uint> mapItemsetToPositionInAll;
	map<uint, uint> mapSingletonToBlkID;
	set<uint> active_blk;
	vector<uint> boundary;
private:
};
#endif /* BLOCKPAGE_H_ */
