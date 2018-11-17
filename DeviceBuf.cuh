/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEVICEBUF_CUH_
#define DEVICEBUF_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "Global.h"
#include "PageTmp.h"
#include "PagesetTmp.h"
#include "Kernel.cuh"

using namespace std;


class DeviceMemory {
public:
	//constructor with no parameter
	DeviceMemory() :  numOfCandidate(0) {
	}
	~DeviceMemory() {
	}

	void initDeviceMemory(int _numOfCandidate, int _verticalListLength) {
		numOfCandidate = _numOfCandidate;
		verticalListLength = _verticalListLength;
		cudaError_t cudaStatus;

		// allocating the memory for two buffers;
		// one stores the bitsets regarding to frequent singletons, and the other stores the supports
		cudaStatus = cudaMalloc((void**) &sourceVerticalList, sizeof(uint) * numOfCandidate * verticalListLength);

		if (cudaStatus != cudaSuccess) {
			cerr  << "initDeviceMemory():: Error in cudaMalloc for sourceVerticalList"	<< endl;
		}
		cudaStatus = cudaMalloc((void**) &supportList, sizeof(uint) * MAX_NUM_CANDIDATE);
		if (cudaStatus != cudaSuccess) {
			cerr << "initDeviceMemory():: Error in cudaMalloc for supportList"	<< endl;
		}
		// initializing the memory for supportList
		cudaStatus = cudaMemset(supportList, 0, sizeof(uint) * MAX_NUM_CANDIDATE);
		if (cudaStatus != cudaSuccess) {
			cerr << "initDeviceMemory():: Error in cudaMemset for supportList" << endl;
		}
 
	}


	void setNumOfCandidates(int _numOfCandidate) {
		numOfCandidate = _numOfCandidate;
	}
	void setNumOfFrequentItemsetsPreviousLevel(
			int _numOfFrequentItemsetsPreviousLevel) {
		numOfFrequentItemsetsPreviousLevel =
				_numOfFrequentItemsetsPreviousLevel;
	}
	void destroyDeviceMemory() {
		cudaError_t cudaStatus;
 		cudaStatus = cudaFree(sourceVerticalList);
		if (cudaStatus != cudaSuccess) {
			cerr << "destroyDeviceMemory():: Error in cudaFree for sourceVerticalList"<< endl;
		}
 		cudaStatus = cudaFree(supportList);
		if (cudaStatus != cudaSuccess) {
			cerr << "destroyDeviceMemory():: Error in cudaFree for supportList"
					<< endl;
		} 
	}

	void initResultVerticalList(int _numOfMaterializedItemsets)
	{
		cudaError_t  cudaStatus;
		numOfMaterializedItemsets = _numOfMaterializedItemsets;
		cudaStatus = cudaMalloc((void**) &resultVerticalList, sizeof(uint) * numOfMaterializedItemsets * verticalListLength);

		if (cudaStatus != cudaSuccess) {
			cerr  << "initResultVerticalList():: Error in cudaMalloc for resultVerticalList"	<< endl;
		}
		cudaStatus = cudaMemset(resultVerticalList, 0, sizeof(uint) * numOfMaterializedItemsets * verticalListLength);
		if (cudaStatus != cudaSuccess) {
			cerr << "initResultVerticalList():: Error in cudaMemset for resultVerticalList" << endl;
		}
	}

	void destroyResultVerticalList()
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaFree(resultVerticalList);
		if (cudaStatus != cudaSuccess) {
			cerr << "destroyResultVerticalList():: Error in cudaFree for resultVerticalList"<< endl;
		}
	}
public:

	// the memory for firstSourceVerticalList and secondSourceVerticalList is allocated on the device memory
	// the buffers for the TFL strategy
	uint* sourceVerticalList;

	// the buffers for the HIL strategy
	uint* resultVerticalList;

	// the array; each element stores the support of a candidate itemset on the specific page
	uint* supportList;

private:
	// it is used as the number of rows
	// it can be adjusted by the user at starting the program
	// we need to consider the case that the number of candidate itemsets generated is larger than numOfCandidate
	int numOfCandidate;

	int numOfFrequentItemsetsPreviousLevel;

	int numOfMaterializedItemsets;

	// verticalListLength represents the number of columns
	// it can be adjusted by the user at starting program
	int verticalListLength;
};

class DeviceMemoryForStream {
public:

	DeviceMemoryForStream() { }

	// initializing the parameters
	DeviceMemoryForStream(int _numOfCandidates, int _lengthOfVerticalList, int _numOfStreams, int _numOfGPUs = 1) {
		numOfCandidates = _numOfCandidates;
		lengthOfVerticalList = _lengthOfVerticalList;
		numOfStreams = _numOfStreams;
	}

	~DeviceMemoryForStream() { }

	// setting the device used now
	cudaError_t setDevice( uint deviceNumber) {
		cudaError_t cudaStatus;
		// Choosing the GPU on a multi-GPUs systems
		cudaStatus = cudaSetDevice(deviceNumber);
		if ( cudaStatus != cudaSuccess ) {
			cerr <<"setDevice():: Error in cudaSetDevice("<<deviceNumber<<")"<<endl;
		}
		return cudaStatus;
	}

	void setVariable(int _numOfCandidates, int _lengthOfVerticalList, int _numOfStreams) {
		numOfCandidates = _numOfCandidates;
		lengthOfVerticalList = _lengthOfVerticalList;
		numOfStreams = _numOfStreams;
 	}

	// creating the streams 
	void createStreams() {
		mPageStreams = new cudaStream_t[numOfStreams];
		for (int loopIdx = 0; loopIdx < numOfStreams; loopIdx++) {
			cudaStreamCreate(&mPageStreams[loopIdx]);
		}
	}

	// allocating the buffers on DM
	void cudaMallocMiningBuffers() {
		cudaError_t cudaStatus;
		MiningBuffers = new DeviceMemory[numOfStreams];
		for (int loopIdx = 0; loopIdx < numOfStreams; loopIdx++) {
			MiningBuffers[loopIdx].initDeviceMemory(numOfCandidates, lengthOfVerticalList);
		}
		cudaStatus = cudaMalloc((void**) &firstIdxSet, sizeof(uint) * MAX_ITEMSET_PABUF * MAX_DEPTH);
		if (cudaStatus != cudaSuccess) {
			cerr << "Error in cudaMalloc for firstIdxSet" << endl;
		}
 	}

	//Realeasing the streams
	void destroyStreams() {
		for (int loopIdx = 0; loopIdx < numOfStreams; loopIdx++) {
			cudaStreamDestroy(mPageStreams[loopIdx]);
		}
	}
	//Realeasing the buffers on DM
	void cudaFreeMiningBuffers() {
		cudaError_t cudaStatus;
		for (int loopIdx = 0; loopIdx < numOfStreams; loopIdx++) {
			MiningBuffers[loopIdx].destroyDeviceMemory();
		}
		if(firstIdxSet) {
			cudaStatus = cudaFree(firstIdxSet);
			if (cudaStatus != cudaSuccess) {
				cerr << "Error in cudaFree for firstIdxSet" << endl;
			}
		}
	}

public:
	// streams
	cudaStream_t* mPageStreams;

	// a set of buffers on the device memory
	DeviceMemory* MiningBuffers;

	uint* firstIdxSet;

private:

	// the number of streams
	int numOfStreams;

	// the number of candidate itemsets on the level (in the apriori algorithm) or the equivalent class (in the eclat algorithm)
	int numOfCandidates;

	// the length of vertical list
	int lengthOfVerticalList;
};
#endif /* DEVICEBUF_CUH_ */
