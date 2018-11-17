/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef PAGETMP_H_
#define PAGETMP_H_
#include "Global.h"
#include <iostream>
using namespace std;

class PageTmp
{
public:
	// constructor with no parameter
	PageTmp() {	}

	// allocating a page
	void initPage(int _pageID, int _verticalListLength, int _numOfCandidate){
		pageID = _pageID;
		verticalListLength = _verticalListLength;
		numOfCandidate = _numOfCandidate;

		cudaError_t cudaStatus;
		cudaStatus = cudaHostAlloc( (void**)&dataForPage,
									 sizeof(uint) * numOfCandidate * verticalListLength,
									 cudaHostAllocDefault);
		if(cudaStatus != cudaSuccess) {
			cerr<<"PageTmp():: Error in cudaHostAlloc for dataForPage"<<endl;
		}


		memset(dataForPage, 0, sizeof(uint)*verticalListLength*numOfCandidate);

	}
	// Realeasing a page
	void destroyPage()	{
		cudaError_t cudaStatus;
		cudaStatus = cudaFreeHost(dataForPage);
		if(cudaStatus != cudaSuccess) {
			cerr<<"PageTmp():: Error in cudaFreeHost for dataForPage" <<endl;
		}
	}
	// copying bitvector of a candidate itemset from TB to TBP
	void updateDataFromWholeArray(int idxOfCandidate, uint* wholeArr, int numOfPages, int wholeArrLength) {
		int startIdx = pageID * verticalListLength;
		if (wholeArrLength < verticalListLength) {
			memcpy( dataForPage + (idxOfCandidate * verticalListLength),
				    wholeArr + startIdx,
					sizeof(uint) * wholeArrLength);
		}
		else {
			if ((wholeArrLength % verticalListLength) == 0) {
				memcpy(dataForPage + (idxOfCandidate * verticalListLength), wholeArr + startIdx, sizeof(uint)*verticalListLength);
			}
			else {
				if((numOfPages-1) == pageID) {
					int remain = wholeArrLength % verticalListLength;
					memcpy(dataForPage + (idxOfCandidate * verticalListLength), wholeArr + startIdx, sizeof(uint)*remain);
				}
				else {
					memcpy(dataForPage+(idxOfCandidate*verticalListLength), wholeArr + startIdx, sizeof(uint)*verticalListLength);
				}
			}
		}
	}
	void updateData(int tailIdx, int candidateIdx, uint* pageArr) {
		int tailOffset = tailIdx * verticalListLength;
		int startOffset = candidateIdx * verticalListLength;
		memcpy(this->dataForPage+(tailOffset), pageArr+(startOffset), sizeof(uint)*verticalListLength);
	}

	void setNumOfCandidate(int _numOfCandidate) {
		numOfCandidate = _numOfCandidate;
	}
	~PageTmp() { }


public:
	// numOfCandidate represents the number of candidates
	// it is used as the number of rows of a buffer on CPU memory
	int numOfCandidate;

	// verticalListLength is the fixed length of each Page_tmp
	// it is used as the number of columns of a buffer on CPU memory
	int verticalListLength;

	// pageID is the ID of pages
	int pageID;

	uint* dataForPage;

private:
};


#endif /* PAGETMP_H_ */
