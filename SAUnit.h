/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef SAUNIT_H_
#define SAUNIT_H_
#include "Global.h"
#include <vector>
using namespace std;

class SAUnit {
public:
	SAUnit() {}
	~SAUnit() {}
	void initSAUnit(int _maxNumCandidates)
	{
		cudaError_t cudaStatus;
		this->maxNumCandidates = _maxNumCandidates;
		cudaStatus = cudaHostAlloc((void**)&supportArr, sizeof(uint)*this->maxNumCandidates, cudaHostAllocDefault);
		if(cudaStatus != cudaSuccess) {
			cerr <<"initSAUnit():: Error in cudaHostAlloc for supportArray" << endl;
		}
		memset(supportArr, 0, sizeof(uint) * this->maxNumCandidates);
	}
	void destroySAUnit()
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaFreeHost(this->supportArr);
		if(cudaStatus != cudaSuccess) {
			cerr <<"destroySAUnit():: Error in cudaFreeHost for supportArray" << endl;
		}
	}
public:
	// the maximum number of candidate itemsets in an iteration
	int maxNumCandidates;
	// the buffer for storing the partial supports
	uint* supportArr;

private:

};
class SAUnitVector {
public:
	SAUnitVector(){}
	~SAUnitVector(){}
	void initSAUnitVector(int _numOfPages, int _maxNumCandidates) {
		this->numOfPages = _numOfPages;
		this->maxNumCandidates = _maxNumCandidates;
		// allociting SAUnit for using async copying the results
		for(int pageIdx = 0; pageIdx < numOfPages; pageIdx++) {
			SAUnit* sa = new SAUnit();
			sa->initSAUnit(maxNumCandidates);
			SAVec.push_back(sa);
		}
	}
	void destroySAUnitVector() {
		for (int pageIdx = 0; pageIdx < numOfPages; pageIdx++)
		{
			SAVec[pageIdx]->destroySAUnit();
			delete SAVec[pageIdx];
		}
		SAVec.clear();
	}
public:
	// the number of pages
	int numOfPages;

	// the maximum number of candidate itemsets in an iteration
	int maxNumCandidates;
	// the buffer for storing the partial supports
	vector<SAUnit*> SAVec;
private:

};


#endif /* SAUNIT_H_ */


