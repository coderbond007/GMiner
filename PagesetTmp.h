/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#ifndef PAGESETTMP_H_
#define PAGESETTMP_H_
#include <vector>
#include "PageTmp.h"

using namespace std;
class PagesetTmp
{
public:
	// constructor with no parameter
	PagesetTmp(){}
	~PagesetTmp(){}

	void initPageset(int _numOfPages, int _numOfCandidates, int _verticalListLength)
	{
		cudaError_t cudaStatus;
		numOfPages = _numOfPages;
		numOfCandidates = _numOfCandidates;
		verticalListLength = _verticalListLength;
		for(int pageIdx = 0; pageIdx < numOfPages; pageIdx++)
		{
			PageTmp* p = new PageTmp();

			// creating an object of PageTmp by cudaHostAlloc()
			p->initPage(pageIdx, verticalListLength, numOfCandidates);
			pageVector.push_back(p);
		}

	}

	void destroyPageset()
	{
		cudaError_t cudaStatus;
		int pageVecSiz = pageVector.size();
		for(int pageVecIdx = pageVecSiz - 1; pageVecIdx > -1; pageVecIdx--)
		{
			pageVector[pageVecIdx]->destroyPage();
			//releasing the memory of pageVec[pageVecIdx] by cudaFreehost if that memory is allocate by cudaHostAlloc()
			delete pageVector[pageVecIdx];
		}
		pageVector.clear();
	}

	void setNumOfCandidate(int _numOfCandidate)
	{
		numOfCandidates = _numOfCandidate;
	}
public:
	// the number of pages
	// it is not changed until the end of program
	int numOfPages;

	// the number of candidates
	// it is the same as the number of rows in each element of pageVector
	int numOfCandidates;

	// the length of verticalList
	// it is adjusted by the user at starting program
	int verticalListLength;

	vector<PageTmp*> pageVector;
private:
};



#endif /* PAGESETTMP_H_ */
