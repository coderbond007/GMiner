/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef MATERIALIZATION_CUH_
#define MATERIALIZATION_CUH_
#include <string>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>

#include <boost/unordered_map.hpp>
#include "item.h"
#include "Global.h"
#include "PageTmp.h"
#include "PagesetTmp.h"
#include "Kernel.cuh"
#include "VerticalDatabase.h"
#include "itemsetUnit.h"
#include "BlockPage.h"

using namespace std;

class Materialization {
public:
	Materialization() {}

	~Materialization() {}

	void initSystemConfig(int _numOfPage, int _numOfStreams, int _numOfGPUs)
	{
		numOfPages = _numOfPage;
		numOfStreams = _numOfStreams;
		numOfGPUs = _numOfGPUs;
	}

	void initMaterialization(verticalDatabase& vd, int _verticalListLength, int _fragmentWidth, double _errorTolerant = 0.1)
	{

		indexCnt = 0;
		fragmentWidth = _fragmentWidth;
		numOfFragment = (vd.allitems.size() - 1)/_fragmentWidth + 1;
		verticalListLength = _verticalListLength;
		mapFrequentSingletons = vd.mapFrequentSingletons;
		minsupAbsolute = vd.minSupAbsolute;
		probablityOfSingletons = vd.probabilityOfSingletons;
		errorTolerant = _errorTolerant;
		minsupRelative = (double) vd.minSupRelative;
		rb_mgr = new rowBlockManager();
		rb_mgr->initRowBlockManager(true, vd.allitems,fragmentWidth);
		int boundaryCnt = 0;

		for (int idx = 0; idx < numOfFragment; idx++) {
			fragmentSize.push_back(fragmentWidth);
		}
		for(int idx = 0; idx < this->fragmentSize.size(); idx++)
		{
 			boundaryCnt += this->fragmentSize[idx];// + 1;
			boundary.push_back(boundaryCnt);
		}

		 vector <uint> v;

		int fragmentNum = 0;
		int boundaryIdx = 0;
		for (int idx = 0; idx < vd.allitems.size();idx++) {

			if(vd.allitems[idx].id < boundary[boundaryIdx]) {
				v.push_back(vd.allitems[idx].id);
				if(idx == vd.allitems.size()-1)
				{
					sort(v.rbegin(),v.rend());
					this->setEnumeration(v, fragmentNum);
 				}
			}
			else
			{
 				sort(v.rbegin(),v.rend());
				this->setEnumeration(v, fragmentNum);
				fragmentNum++;
				v.clear();
				boundaryIdx++;
				idx--;
			}
		}

	}

	void initMaterialization_prob(verticalDatabase& vd, int _fragmentWidth, double _errorTolerant = 0.1)
	{
		indexCnt = 0;
		numOfFragment = 0;
		fragmentWidth = _fragmentWidth;
		mapFrequentSingletons = vd.mapFrequentSingletons;
		minsupAbsolute = vd.minSupAbsolute;
		probablityOfSingletons = vd.probabilityOfSingletons;
		errorTolerant = _errorTolerant;
		minsupRelative = (double) vd.minSupRelative;

		// build boundary for dividing ranges of the indexes of frequent 1-itemsets
		this->boundarySelection();
		int boundaryCnt = 0;
		for(int idx = 0; idx < this->fragmentSize.size(); idx++)
		{
 			boundaryCnt += this->fragmentSize[idx];// + 1;
			boundary.push_back(boundaryCnt);
		}


		// enumerating all the subsets of hot items
		vector <uint> v;
		int v_idx = 0;
		int fragmentNum = 0;
		int boundaryIdx = 0;

		for (int idx = 0; idx < vd.allitems.size();idx++)
		{

			if(vd.allitems[idx].id < boundary[boundaryIdx])
			{

				v.push_back(vd.allitems[idx].id);
				if(idx == vd.allitems.size()-1)
				{
					sort(v.rbegin(),v.rend());
					this->setEnumeration(v, fragmentNum);
 				}
			}
			else
			{
 				sort(v.rbegin(),v.rend());
				this->setEnumeration(v, fragmentNum);
				fragmentNum++;
				v.clear();
				boundaryIdx++;
				idx--;
			}
		}

	}

	void buildMaterialization(PagesetTmp* initialPageset, DeviceMemoryForStream* DM, vector<SAUnit*> _saVec)
	{
		cout <<"Building the buffers of TB and SA with respect to the HIL strategy. [";
		Timer startBuildBuffer = getCurrentTime();
		cudaError_t cudaStatus;
		materialization = new PagesetTmp;
		materialization->initPageset( initialPageset->numOfPages,
									  indexCnt,
									  initialPageset->verticalListLength);

		uint* _idxSetFirstSource = new uint[indexCnt*MAX_DEPTH];
		memset( _idxSetFirstSource, 0, sizeof(uint)*indexCnt*MAX_DEPTH);

		uint* supportArr = new uint[indexCnt];
		memset(supportArr, 0, sizeof(uint)*indexCnt);

		Timer endBuildBuffer = getCurrentTime();
		cout <<(endBuildBuffer-startBuildBuffer)<<" sec]"<<endl;
		cout <<"Recording the position sets with respect to the itemsets pre-computed. [";
		Timer startRecordPos = getCurrentTime();
		long long int srcIdxStartOffset = 0;
		for (int candidateIdx = 0; candidateIdx < indexCnt; candidateIdx++)
		{
			int count=0;
			srcIdxStartOffset = (long long int)candidateIdx * (long long int)MAX_DEPTH;
			srcIdxStartOffset++;
			for(int lenIdx = 0; lenIdx < MaterializedItemset[candidateIdx].size(); lenIdx++)
			{
				_idxSetFirstSource[srcIdxStartOffset++] = MaterializedItemset[candidateIdx][lenIdx];
				count++;
			}
			long long int candidateStartOffset = (long long int) candidateIdx * (long long int) MAX_DEPTH;
			_idxSetFirstSource[candidateStartOffset ] = count;
		}
		Timer endRecordPos = getCurrentTime();
		cout <<(endRecordPos-startRecordPos)<<" sec]"<<endl;
		cout <<"Building the buffers on GPU device memory with respect to the HIL strategy. [";
		Timer startBuildDM = getCurrentTime();
		for(uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {
			cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
			if(cudaStatus != cudaSuccess) {
				cerr <<"buildMaterialization():: Error in setting the GPU device["<<gpuIdx<<"]" << endl;
			}
			for (int streamIdx = 0; streamIdx < numOfStreams; streamIdx++)
			{
				DM[gpuIdx].MiningBuffers[streamIdx].initResultVerticalList(indexCnt);
			}
		}
		Timer endBuildDM = getCurrentTime();
		cout <<(endBuildDM-startBuildDM)<<" sec" << endl;
		Timer start = getCurrentTime();
		cout <<"Generating pre-computations. [";
		generatePrecomputation_mod
							   ( DM,                                  //
								initialPageset->pageVector,
								this->materialization->pageVector,
								_saVec,
								_idxSetFirstSource,
								supportArr,
								numOfPages,
								initialPageset->numOfCandidates,
								this->indexCnt,
								initialPageset->verticalListLength,
								MAX_DEPTH);

		Timer end = getCurrentTime();
		cout <<(end-start)<<" sec]" << endl;

		if(_idxSetFirstSource){
			delete[] _idxSetFirstSource;
			_idxSetFirstSource = NULL;
		}
		if(supportArr) {
			delete[] supportArr;
			supportArr = NULL;
		}


	}




	void generatePrecomputation_mod ( DeviceMemoryForStream* DM,
									  vector<PageTmp*> sourceVector,
									  vector<PageTmp*> resultVector,
									  vector<SAUnit*> saVec,
									  uint* idxSet,
									  uint* support,
									  int numOfPages,
									  int numOfFrequentItemsetsPreviousLevel,
									  int numOfCandidates,
									  int lenOfList,
									  int depthMax)
	{

		cudaError_t cudaStatus;
		uint tmpNumOfGPUs = numOfGPUs;
		numOfGPUs = 1;
		uint RUN_MAX_BLK = 0 ;
		if (numOfCandidates < MAX_ITEMSET_PABUF) {
			RUN_MAX_BLK = (numOfCandidates - 1) / numOfGPUs + 1;
		}
		else {
			RUN_MAX_BLK = MAX_ITEMSET_PABUF;
		}

		int numOfIAPs = (numOfCandidates - 1) / RUN_MAX_BLK + 1;
		bool inOuterLoop = true;
		cout <<"RUN_MAX_NLK - " << RUN_MAX_BLK << ", numOfIAPs - " << numOfIAPs << endl;
		for(uint outerLoopIdx = 0; outerLoopIdx < numOfIAPs; outerLoopIdx += numOfGPUs)
		{
			//replicate one of outer join operands to each GPU
			for (uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {
				if((outerLoopIdx + gpuIdx) > numOfIAPs) {
					inOuterLoop = false;
					break;
				}
				cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "generatePrecomputation_mod():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
				}
				uint* IAPAddr = idxSet + ( ( outerLoopIdx + gpuIdx ) * RUN_MAX_BLK	* MAX_DEPTH ) ;
				cout <<"generatePrecomputation_mod():: offset(idxSet) - " <<( ( outerLoopIdx + gpuIdx ) * RUN_MAX_BLK	* MAX_DEPTH) << endl;
				cudaStatus = cudaMemcpy( DM[gpuIdx].firstIdxSet,
									     IAPAddr,
									     sizeof(uint) * MAX_DEPTH * RUN_MAX_BLK,
									     cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "generatePrecomputation_mod():: Error in cudaMemcpy for firstIdxSet of the GPU device["<< gpuIdx << "]" << endl;
				}
			}

			if(inOuterLoop == false) {
				break;
			}

			for(uint pageIdx = 0; pageIdx < numOfPages; pageIdx++)
			{
				// replicating an inner join operand to all the GPUs
				for (uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++)
				{
					cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
					if (cudaStatus != cudaSuccess) {
						cerr 	<< "generatePrecomputation_mod():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
					}
					uint streamOffset = pageIdx % numOfStreams;

					cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
													sourceVector[pageIdx]->dataForPage,
													sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
													cudaMemcpyHostToDevice,
													DM[gpuIdx].mPageStreams[streamOffset]);

					if (cudaStatus != cudaSuccess) {
						cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
					}
					Kernel_Materialization<<<RUN_MAX_BLK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
					( DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
					  DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
					  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
					  DM[gpuIdx].firstIdxSet,
					  numOfCandidates,
					  lenOfList,
					  depthMax,
					  ((outerLoopIdx+gpuIdx) * RUN_MAX_BLK),
					  RUN_MAX_BLK);

					cudaStatus = cudaMemcpyAsync(resultVector[pageIdx]->dataForPage + (outerLoopIdx + gpuIdx) * RUN_MAX_BLK * lenOfList,
												 DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
												// sizeof(uint) * numOfCandidates * lenOfList,
												 sizeof(uint) * RUN_MAX_BLK * lenOfList,
												 cudaMemcpyDeviceToHost,
												 DM[gpuIdx].mPageStreams[streamOffset]);

					if (cudaStatus != cudaSuccess)
					{
						cerr << "buildMaterialization():: Error in cudaMemcpyAsync for resultVector" << endl;
					}


					uint* partialSupportArr = saVec[pageIdx]->supportArr + (outerLoopIdx + gpuIdx) * RUN_MAX_BLK;
					cudaStatus = cudaMemcpyAsync( partialSupportArr,
												  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
												  sizeof(uint) * RUN_MAX_BLK,
												  cudaMemcpyDeviceToHost,
												  DM[gpuIdx].mPageStreams[streamOffset]);
					 if (cudaStatus != cudaSuccess) {
						 cerr << "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for supportArr" << endl;
					 }
					 if(pageIdx % numOfStreams == (numOfStreams-1))
					 {
						cudaStatus = cudaDeviceSynchronize();
						if(cudaStatus != cudaSuccess) {
							cerr <<"Error in cudaDeviceSynchronize"<<endl;
						}
					 }
				}

			}

		}

		 for (uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {
			 // set the GPU device used at a time
			 cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
			 if ( cudaStatus != cudaSuccess) {
				 cerr <<"supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["<<gpuIdx<<"]"<<endl;
			 }
			 cudaStatus = cudaThreadSynchronize();
			 if(cudaStatus != cudaSuccess) {
				 cerr <<"supportCountingIndexDelivery():: Error in cudaThreadSynchronize()" << endl;
			 }
		 }

		 for (int pageIdx = 0; pageIdx < numOfPages; pageIdx++) {
			 for (int candidateIdx = 0; candidateIdx < numOfCandidates; candidateIdx++) {
				 support[candidateIdx] += saVec[pageIdx]->supportArr[candidateIdx];
			 }
		 }
		numOfGPUs = tmpNumOfGPUs;
	}



	void destroyMaterialization(DeviceMemoryForStream* DM)
	{
		cudaError_t cudaStatus;
		for(uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {
			cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
			if(cudaStatus != cudaSuccess) {
				cerr <<"buildMaterialization():: Error in setting the GPU device["<<gpuIdx<<"]" << endl;
			}
			for (int streamIdx = 0; streamIdx < numOfStreams; streamIdx++)
			{
				DM[gpuIdx].MiningBuffers[streamIdx].destroyResultVerticalList();
			}
		}
	}
	void boundarySelection()
	{
		double boundaryProbability = errorTolerant * minsupRelative;
		int boundaryCnt  = 0;
		double probabilityProdoct = 1.0;
		for(int idx = probablityOfSingletons.size()-1; idx > -1 ; idx--)
		{
			probabilityProdoct *= probablityOfSingletons[idx];
			if(boundaryProbability <= probabilityProdoct)
			{
				boundaryCnt++;
				if(idx == 0)
				{
					fragmentSize.push_back(boundaryCnt);
				}
			}
			else
			{
				fragmentSize.push_back(boundaryCnt);
				boundaryCnt = 0;
				probabilityProdoct = 1.0;
				idx++;
			}
		}
		int sum=0;
		sort(fragmentSize.begin(), fragmentSize.end());
		for(int idx = 0; idx < fragmentSize.size();idx++)
		{
			sum+=fragmentSize[idx];
		}

	}


private:
	//void setEnumeration(int* arr, int len) {
	void setEnumeration(vector<uint> arr, int _fragmentNum) {
 		idxType mm;

		if(arr.size()==1)
		{
 			mm[arr] = indexCnt;
 			positionDictionary[arr] = indexCnt;
 			indexCnt++;
 			MaterializedItemset.push_back(arr);
 			return;
		}
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
			mm[v] = indexCnt;
			positionDictionary[v] = indexCnt;
			indexCnt++;
			MaterializedItemset.push_back(v);
		}
	}

public:
	vector<uint> hotItem;

	vector<uint> coldItem;
	// boundary will used to partition a candidate itemset into a number of CIPs
	vector<uint> boundary;

	// the following variables are used for boundary selection
	vector<double> probablityOfSingletons;
	vector<int> fragmentSize;
	double minsupRelative;
	double errorTolerant;
	int bestNumOfMaterialization;

	vector<vector<uint> > MaterializedItemset;

	map<vector<uint>, uint>   positionDictionary;
	map<int, int> mapFrequentSingletons;
	PagesetTmp* materialization;


	// the length of vertical list
	int verticalListLength;

	// the number of element
	// 2^(the number of (hotItem-1)/width + 1)
	int numOfElement;

	// the width of fragmentation
	int fragmentWidth;

	int numOfFragment;

	int indexCnt;
	int numOfPages;
	int numOfStreams;
	int numOfGPUs;

	int minsupAbsolute;

	rowBlockManager* rb_mgr;
};




#endif /* MATERIALIZATION_CUH_ */
