/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef FRAMEWORK_CUH_
#define FRAMEWORK_CUH_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <boost/unordered_map.hpp>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#include <fstream>

#include "SAUnit.h"

#include "PagesetTmp.h"
#include "VerticalDatabase.h"
#include "Global.h"
#include "DeviceBuf.cuh"
#include "Materialization.cuh"
#include "item.h"
#include "BlockPage.h"

using namespace std;

class Framework {
public:

	// constructor with no parameter
	Framework() {
	}

	~Framework() {
	}
 
	void initFramework ( verticalDatabase& vd,
			     inputParameter* input,
			     int _maxDepth = 30) {
		minSingleton = 0;
		this->mapFrequentSingletons = vd.mapFrequentSingletons;
		this->numOfCandidates = vd.allitems.size();
		this->numOfSingletons = vd.allitems.size();
		this->verticalListLength = input->verticalListLength;//_verticalListLength;
		this->numOfTransactions = vd.tidCount;
		this->wholeVerticalListLength =	(numOfTransactions % 32 == 0 ? (numOfTransactions / 32) : (numOfTransactions / 32 + 1));
		this->numOfPages = ( wholeVerticalListLength % verticalListLength == 0 ? (wholeVerticalListLength / verticalListLength) : (wholeVerticalListLength / verticalListLength + 1));
		this->minsupAbsoulte = vd.minSupAbsolute;
		this->numOfStreams = input->numOfStreams;
		this->maxDepth = _maxDepth;
		this->isOutput = input->isWriteOutput;
		if(this->isOutput == 1) {
			fout.open(input->outputPath);
		}
		// a variable for the multiple GPUs system
		this->numOfGPUs = input->numOfGPUs;

		// initialize the virables for parallelization of recordIndexes()
		this->numOfCPUs = sysconf(_SC_NPROCESSORS_ONLN);

		// a variable used in Apriori-based approach
		this->level = 1;

		// a varialble used in Materialization
		this->isMaterialization = input->isMaterialization;//_isMaterialization;

		// initialize the buffers on MM
		this->initBufferMM(vd);

		// initialize tries
		this->initTrie(vd);

		// initialize the buffers on DM
		this->initBufferDM(numOfCandidates, verticalListLength, numOfStreams, numOfGPUs);

		// generate pre-computations in the HIL strategy
		if (isMaterialization == true) {
			materialization = new Materialization();
			materialization->initSystemConfig(this->numOfPages, this->numOfStreams, this->numOfGPUs);
			materialization->initMaterialization(vd, verticalListLength, input->sizeOfFragments);
			materialization->buildMaterialization(this->initialPageset,	this->DM, this->saVec->SAVec);
			this->initialPageset->destroyPageset();
		} else {
			this->mapSingletonIndex = vd.mapSingletonIndex;
		}


	}

	void printFrameworkStatus() {
		cout << "initFramework():: numOfCandidates - " << numOfCandidates
				<< ", verticalListLength - " << verticalListLength
				<< ", numOfTransactions - " << numOfTransactions
				<< ", wholeVerticalListLength - " << wholeVerticalListLength
				<< ", numOfPages - " << numOfPages << ", numOfStreams - "
				<< numOfStreams << endl;

	}

	void initBufferDM(int _numOfCandidates, int _verticalListLength, int _numOfStreams, int _numOfGPUs) {

		cout <<"Allocating the buffers on GPU device memory. [";
		Timer startInitBufferDM = getCurrentTime();
		this->DM = new DeviceMemoryForStream[_numOfGPUs];
		for (uint loopIdx = 0; loopIdx < _numOfGPUs; loopIdx++) {
			this->DM[loopIdx].setDevice(loopIdx);
			this->DM[loopIdx].setVariable(_numOfCandidates, _verticalListLength, 	_numOfStreams);
			this->DM[loopIdx].createStreams();
			this->DM[loopIdx].cudaMallocMiningBuffers();
		}
		Timer endInitBufferDM = getCurrentTime();
		cout <<(endInitBufferDM-startInitBufferDM)<<" sec]"<<endl;

	}

	void initBufferMM(verticalDatabase& vd) {
		cudaError_t cudaStatus;
		for(int idx = 0; idx < CPU_THREAD; idx++) {
			vector<uint> tmp_v;
			positionVector.push_back(tmp_v);
		}

		// Computing the number of blocks
		uint blk_num = (wholeVerticalListLength-1) / verticalListLength + 1;
		wholeVerticalListLength = blk_num * verticalListLength;
		wholeVerticalList = new uint*[numOfCandidates];
		for (int candidateIdx = 0; candidateIdx < numOfCandidates; candidateIdx++) {
			wholeVerticalList[candidateIdx] = new uint[wholeVerticalListLength];
			for (int lenIdx = 0; lenIdx < wholeVerticalListLength; lenIdx++) {
				wholeVerticalList[candidateIdx][lenIdx] = 0;
			}
		}

		Timer startBuildVerticalDB, endBuildVerticalDB;
		startBuildVerticalDB = getCurrentTime();
		//convert tids of vd into bitsets
		cout <<"Transforming vertical format with bit vectors from horizontal format. [";
		for (int idx = 0; idx < vd.allitems.size(); idx++) {
			vector<unsigned> itemsetVector;
			itemsetVector.push_back(vd.allitems[idx].id);
			frequentItemsets.push_back(itemsetVector);
			uint* tmp = wholeVerticalList[idx];
			for (int arrayIdx = 0; arrayIdx < vd.allitems[idx].transactions.size();  arrayIdx++) {
				setbit(tmp, vd.allitems[idx].transactions[arrayIdx], 1);
			}
			if(isOutput == 1) {
				if (vd.allitems[idx].support > 0){

					map<int,int>::iterator m_it;
					m_it = mapFrequentSingletons.find(vd.allitems[idx].id);
					//fout << m_it->second <<" ";

					fout << m_it->second <<" ("<<(float)vd.allitems[idx].support/(float)numOfTransactions<<")"<<endl;
					//cout << vd.allitems[idx].id<<" ("<<(float)vd.allitems[idx].support/(float)numOfTransactions<<")"<<endl;
				}
			}
		}
		endBuildVerticalDB = getCurrentTime();
		cout <<(endBuildVerticalDB - startBuildVerticalDB) <<" sec]"<<endl;


		// Building the buffers with respect to GMiner
		cout <<"Building the buffers of TB and SA with respect to the TFL strategy. [";
		Timer startBuildingBuffers = getCurrentTime();
		initialPageset = new PagesetTmp();
		initialPageset->initPageset(numOfPages, numOfCandidates, verticalListLength);

		long long int idxSetArraySize = (long long int)MAX_NUM_CANDIDATE * (long long int)maxDepth;
 		idxSetArray = new uint[idxSetArraySize];
		memset(idxSetArray, 0, sizeof(uint) * idxSetArraySize);
		cudaStatus = cudaHostAlloc(  (void**) &supportArray,
									 sizeof(uint) * MAX_NUM_CANDIDATE,
									 cudaHostAllocDefault);
		if (cudaStatus != cudaSuccess) {
			cerr << "initFramework():: Error in cudaHostAlloc for supportArray" << endl;
		}
		memset(supportArray, 0, sizeof(uint) * MAX_NUM_CANDIDATE);
		this->saVec = new SAUnitVector();
		this->saVec->initSAUnitVector(numOfPages, MAX_NUM_CANDIDATE);

		for (int pageIdx = 0; pageIdx < numOfPages; pageIdx++) {
			for (int candidateIdx = 0; candidateIdx < numOfCandidates; candidateIdx++) {
				uint* oneDimVerticalList = wholeVerticalList[candidateIdx];
				initialPageset->pageVector[pageIdx]->updateDataFromWholeArray(
														candidateIdx,
														oneDimVerticalList,
														numOfPages,
														wholeVerticalListLength);

			}
		}
		Timer endBuildingBuffers = getCurrentTime();
		cout <<(endBuildingBuffers-startBuildingBuffers)<<" sec]"<<endl;
  		cntArr = new int[CPU_THREAD];
		this->cleanWholeVerticalList();

	}

	void initTrie(verticalDatabase& vd) {
		cout <<"Allocating the buffers with respect to candidate itemsets. [";
		Timer startInitTrie = getCurrentTime();
		// memory optimization by the pingpong execution
		candidate_list_length = numOfCandidates;
		candidate_list_width = 1;
		candidate_list = new unsigned short*[maxDepth];
		for (int idx = 0; idx < maxDepth; idx++) {
			candidate_list[idx] = new unsigned short[MAX_NUM_CANDIDATE];
			memset((void*) candidate_list[idx], 0, sizeof(unsigned short) * MAX_NUM_CANDIDATE);
		}
		validate = new unsigned short[MAX_NUM_CANDIDATE];
		p_boundary_list = new vector<int>;

		memset((void *) (validate), 0, sizeof(unsigned short) * MAX_NUM_CANDIDATE);
		for (int idx = 0; idx < vd.allitems.size(); idx++) {
			candidate_list[0][vd.allitems[idx].id] = (unsigned short)vd.allitems[idx].id;
			validate[idx] = 1;
		}
		p_boundary_list->push_back(candidate_list_length);

		// initialize the buffers for the pingpong execution
		new_candidate_list = new unsigned short*[maxDepth];
		for (int idx = 0; idx < maxDepth; idx++) {
			new_candidate_list[idx] = new unsigned short[MAX_NUM_CANDIDATE];
			memset((void*) new_candidate_list[idx], 0, sizeof(unsigned short) * MAX_NUM_CANDIDATE);
		}
		new_p_boundary_list = new vector<int>;
		Timer endInitTrie = getCurrentTime();
		cout <<(endInitTrie-startInitTrie)<<" sec]"<<endl;
	}

	void printVec(vector<uint> v) {
		cout << "vv - ";
		for (int idx = 0; idx < v.size(); idx++) {
			cout << v[idx] << " ";
		}
		cout << endl;
	}
	void printidxSetArray(int _numOfCandidates, int _maxDepth) {
		cout << "printidxSetArray()" << endl;
		for (int idx = 0; idx < _numOfCandidates; idx++) {
			cout << "candidateIdx - " << idx << endl;
			for (int lenIdx = 0; lenIdx < _maxDepth; lenIdx++) {
				cout << idxSetArray[idx * _maxDepth + lenIdx] << " ";
			}
			cout << endl;
		}
	}
			
	void printOutput(unsigned short** candidate_list, uint* supportArr, unsigned short* validate) {
		map<int,int>::iterator m_it;
		cout <<"candidate_list_length - " << candidate_list_length <<", candidate_list_width - " << candidate_list_width << ", numOfTransactions - " << numOfTransactions << endl;
		for(long long int candidateIdx = 0; candidateIdx < candidate_list_length; candidateIdx++ ) {
			for(long long int lenIdx = 0; lenIdx < candidate_list_width; lenIdx++) {
				if(validate[candidateIdx] == 1) {
					m_it = mapFrequentSingletons.find(candidate_list[lenIdx][candidateIdx]);
					fout << m_it->second <<" ";
					//cout << candidate_list[lenIdx][candidateIdx] <<" ";
				}
			}
			if(validate[candidateIdx] == 1) {
				fout<<"("<<(float)supportArr[candidateIdx] / (float)numOfTransactions<<")"<<endl;
			}

		}
	}
	// Performing an iteration composed of candidate generation using CPUs and support counting using GPUs
	int nextLevel() {
		level++;
		int numOfCandidatesForLevel = -1;
		int numOfFrequentForLevel = -1;

		Timer startSupportCounting, endSupportCounting;
		Timer startCandidateGeneration = getCurrentTime();


		memset(supportArray, 0, sizeof(uint) * candidate_list_length);

		// candidate generation
		if (candidate_list_width % 2 == 1) {
			candidateGenerationTrie( candidate_list,
									 new_candidate_list,
									 p_boundary_list,
									 new_p_boundary_list,
									 validate,
									 idxSetArray);
		} else {
			candidateGenerationTrie( new_candidate_list,
									 candidate_list,
									 new_p_boundary_list,
									 p_boundary_list,
									 validate,
									 idxSetArray);
		}


		// Storing the position sets of candidate itemsets in the current iteration
		if(isMaterialization == true) {
			Timer startRecord, endRecord;
			startRecord = getCurrentTime();
			cout <<"Recording the positionsets in the HIL strategy. [";
			parallelRecordIndexes_fixed_width();
			endRecord = getCurrentTime();
			cout <<(endRecord-startRecord)<<" sec]" << endl;

		}

		Timer endCandidateGeneration = getCurrentTime();
		elapsedTimeCandidateGeneration += (endCandidateGeneration - startCandidateGeneration);
		numOfCandidatesForLevel = candidate_list_length;

		startSupportCounting = getCurrentTime();
		cout <<"Performing support counting. [";
		if (isMaterialization == false) {
			GMiner_Default
			 ( DM,
 			  this->initialPageset->pageVector,
    		  saVec->SAVec,
    		  idxSetArray,
    		  supportArray,
    		  numOfPages,
     		  this->initialPageset->numOfCandidates,
    		  numOfCandidatesForLevel,
    		  verticalListLength,
    		  level,
    		  maxDepth);
		} else {
			GMiner_Default
			 ( DM,
    		  this->materialization->materialization->pageVector,
    		  saVec->SAVec,
    		  idxSetArray,
    		  supportArray,
    		  numOfPages,
    		  this->materialization->indexCnt,
    		  numOfCandidatesForLevel,
    		  verticalListLength,
    		  level,
    		  maxDepth);
		}
		endSupportCounting = getCurrentTime();
		cout <<(endSupportCounting-startSupportCounting)<< " sec]"<<endl;


		// pruning infrequent itemsets
		if (candidate_list_width % 2 == 0) {
			numOfFrequentForLevel = this->parallelPruningCandidates(new_candidate_list);
		} else {
			numOfFrequentForLevel = this->parallelPruningCandidates(candidate_list);
		}
		// printing the outputs
		if(isOutput == 1) {
			cout <<"Printing frequent itemsets." << endl;
			if(candidate_list_width % 2 == 1) {
				printOutput(candidate_list, supportArray, validate);
			}
			else {
				printOutput(new_candidate_list, supportArray, validate);
			}
		}
		return numOfFrequentForLevel;
	}




	// Performing an iteration by different strategy in the paper
	void GMiner_Default_replicate_positionset(
			DeviceMemoryForStream* DM,
			vector<PageTmp*> sourceVector,
			//SAUnitVector* supportVector,
			vector<SAUnit*> supportVector,
			uint* idxSet,
			uint* support,
			int numOfPages,
			int numOfFrequentItemsetsPreviousLevel,
			int numOfCandidates,
			int lenOfList,
			int depth,
			int depthMax) {

		// if there is no candidate itemset, exit this function
		if (numOfCandidates == 0) {
			return;
		}
		cudaError_t cudaStatus;

		uint firstActiveBlock = 0;
                set<uint>::iterator activeBlkIterator = materialization->rb_mgr-> active_blk.begin();
		firstActiveBlock = *activeBlkIterator;

		cout <<"numOfCandidates - " <<  numOfCandidates <<", numOfGPUs - " << numOfGPUs << endl;




		if(numOfCandidates < MAX_ITEMSET_PABUF)	{
			RUN_MAX_BLK = (numOfCandidates-1) / numOfGPUs + 1;
		}
		else {
			RUN_MAX_BLK = MAX_ITEMSET_PABUF;
		}
		int numOfIAPs = (numOfCandidates - 1) / RUN_MAX_BLK + 1;
		int IAPPos = 0;

		bool inOuterLoop = true;

		cout <<"RUN_MAX_BLK - " << RUN_MAX_BLK << ", numOfIAPs - " << numOfIAPs << endl;


		for(uint outerLoopIdx = 0; outerLoopIdx < numOfIAPs; outerLoopIdx++ ) {

			// replicate one of outer join operand to all gpus
			for(uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {

				cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
				}
				uint* IAPAddr = idxSet + ( ( outerLoopIdx ) * RUN_MAX_BLK	* MAX_DEPTH ) ;
				//cout <<"offset(idxSet) - " <<( ( outerLoopIdx + gpuIdx ) * RUN_MAX_BLK	* MAX_DEPTH) << endl;
				cudaStatus = cudaMemcpy( DM[gpuIdx].firstIdxSet,
									     IAPAddr,
									     sizeof(uint) * MAX_DEPTH * RUN_MAX_BLK,
									     cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in cudaMemcpy for firstIdxSet of the GPU device["<< gpuIdx << "]" << endl;
				}
			}

			// process an inner join operand
			uint numOfPagesEachGPU = numOfPages / numOfGPUs;
			uint remainNumOfPagesEachGPU = numOfPages % numOfGPUs;
 			for(uint pageIdx = 0; pageIdx < numOfPagesEachGPU; pageIdx++)
			{
				// distributing an inner join operand over GPUs
				for (uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++)
				{
					uint pageOffset = pageIdx * numOfGPUs + gpuIdx;


					cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
					if (cudaStatus != cudaSuccess) {
						cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
					}
					uint streamOffset = pageIdx % numOfStreams;
					cout <<"pageIdx - " << pageIdx <<", gpuIdx - " << gpuIdx <<", streamOffset - " << streamOffset<< endl;
					if (isMaterialization == true) {


						uint numOfElementsInFragment = 50;
						uint  fragmentIdx = 0;
						uint fragmentIteration = (numOfFrequentItemsetsPreviousLevel) / numOfElementsInFragment;
						uint fragmentRemain = numOfFrequentItemsetsPreviousLevel % numOfElementsInFragment;


						uint numOfpartialBitmap = numOfFrequentItemsetsPreviousLevel - (materialization->rb_mgr->numOfItemsetsMaterialized * firstActiveBlock );

						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
														//sourceVector[pageIdx]->dataForPage
	     											    sourceVector[pageOffset]->dataForPage + (firstActiveBlock * verticalListLength * materialization->rb_mgr->numOfItemsetsMaterialized),
	     											    // full copy
														//this->blks_buffer_vector[streamOffset],
	     											    //sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
	     											   sizeof(uint) * (numOfpartialBitmap) * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}
						Kernel_HFI<<< RUN_MAX_BLK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
						( DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
						  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
						  DM[gpuIdx].firstIdxSet,
						  numOfCandidates,
						  lenOfList,
						  depthMax,
						  ((outerLoopIdx) * RUN_MAX_BLK),
						  RUN_MAX_BLK
						);
					}
					else {

						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
	     											    sourceVector[pageOffset]->dataForPage,
	     											    sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}

						Kernel_TFL<<<RUN_MAX_BLK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
						( DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
					      DM[gpuIdx].MiningBuffers[streamOffset].supportList,
						  DM[gpuIdx].firstIdxSet,
						  numOfCandidates,
						  lenOfList,
						  depth,
						  depthMax,
						  ((outerLoopIdx) * RUN_MAX_BLK),
						  RUN_MAX_BLK
						);
					}

					uint* partialSupportArr = supportVector[pageOffset]->supportArr + (outerLoopIdx) * RUN_MAX_BLK;
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
 			for(uint pageIdx = numOfGPUs*numOfPagesEachGPU; pageIdx < numOfGPUs*numOfPagesEachGPU + remainNumOfPagesEachGPU; pageIdx++)
			{
				// distributing an inner join operand over GPUs
 				uint gpuIdx = pageIdx % numOfGPUs;
					cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
					if (cudaStatus != cudaSuccess) {
						cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
					}
					uint streamOffset = pageIdx % numOfStreams;

					if (isMaterialization == true) {


						uint numOfElementsInFragment = 50;
						uint  fragmentIdx = 0;
						uint fragmentIteration = (numOfFrequentItemsetsPreviousLevel) / numOfElementsInFragment;
						uint fragmentRemain = numOfFrequentItemsetsPreviousLevel % numOfElementsInFragment;



						uint numOfpartialBitmap = numOfFrequentItemsetsPreviousLevel - (materialization->rb_mgr->numOfItemsetsMaterialized * firstActiveBlock );

						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
														//sourceVector[pageIdx]->dataForPage
	     											    sourceVector[pageIdx]->dataForPage + (firstActiveBlock * verticalListLength * materialization->rb_mgr->numOfItemsetsMaterialized),
	     											    // full copy
														//this->blks_buffer_vector[streamOffset],
	     											    //sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
	     											   sizeof(uint) * (numOfpartialBitmap) * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}
						//cudaThreadSynchronize();
						Kernel_HFI<<< RUN_MAX_BLK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
						( DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
						  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
						  DM[gpuIdx].firstIdxSet,
						  numOfCandidates,
						  lenOfList,
						  depthMax,
						  ((outerLoopIdx) * RUN_MAX_BLK),
						  RUN_MAX_BLK
							//(IAPIdx*numOfGPUs + gpuIdx) * MAX_BLOCK
						);
					}
					else {

						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
	     											    sourceVector[pageIdx]->dataForPage,
	     											    sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}

						Kernel_TFL<<<RUN_MAX_BLK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
						( DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
					      DM[gpuIdx].MiningBuffers[streamOffset].supportList,
						  DM[gpuIdx].firstIdxSet,
						  numOfCandidates,
						  lenOfList,
						  depth,
						  depthMax,
						  ((outerLoopIdx) * RUN_MAX_BLK),
						  RUN_MAX_BLK
						);
					}

					uint* partialSupportArr = supportVector[pageIdx]->supportArr + (outerLoopIdx) * RUN_MAX_BLK;
					cudaStatus = cudaMemcpyAsync( partialSupportArr,
												  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
												  sizeof(uint) * RUN_MAX_BLK,
												  cudaMemcpyDeviceToHost,
												  DM[gpuIdx].mPageStreams[streamOffset]);
					 if (cudaStatus != cudaSuccess) {
						 cerr << "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for supportArr" << endl;
					 }
					// if(pageIdx % numOfStreams == (numOfStreams-1))
					 //{

					 //}
			//	}

			}
			cudaStatus = cudaDeviceSynchronize();
			if(cudaStatus != cudaSuccess) {
				cerr <<"Error in cudaDeviceSynchronize"<<endl;
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
				 support[candidateIdx] += supportVector[pageIdx]->supportArr[candidateIdx];
			 }
		 }
		cout <<"End func"<<endl;
	}

	// the GPU kernel function support counting 1-1
	// the outer join operand - Index Array (IA)
	// the inner join operand - transaction Bitmap (TB)
	// the replication strategy - replicating the inner join operand
	// the performance - the second

	void GMiner_Default(
			DeviceMemoryForStream* DM,
			vector<PageTmp*> sourceVector,
 			vector<SAUnit*> supportVector,
			uint* idxSet,
			uint* support,
			int numOfPages,
			int numOfFrequentItemsetsPreviousLevel,
			int numOfCandidates,
			int lenOfList,
			int depth,
			int depthMax) {

		if (numOfCandidates == 0) {
			return;
		}
		cudaError_t cudaStatus;

		uint firstActiveBlock = 0;

		// finding the necessary blocks
		if (isMaterialization == true) {
			map<uint, uint>::iterator blkIterator = materialization->rb_mgr->mapSingletonToBlkID.find(minSingleton);
			firstActiveBlock = blkIterator->second;//this->minSingleton;
		}

		if(numOfCandidates < MAX_ITEMSET_PABUF)	{
			RUN_MAX_BLK = (numOfCandidates-1) / numOfGPUs + 1;
		}
		else {
			RUN_MAX_BLK = MAX_ITEMSET_PABUF;
		}
		int numOfIAPs = (numOfCandidates - 1) / RUN_MAX_BLK + 1;


		int IAPPos = 0;
 		bool inOuterLoop = true;

		for(uint outerLoopIdx = 0; outerLoopIdx < numOfIAPs; outerLoopIdx += numOfGPUs)
		{
			// replicate one of outer join operand to each GPU
			for(uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++) {
 				if ((outerLoopIdx + gpuIdx) > numOfIAPs) {
 					inOuterLoop = false;
					break;
				}
				cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
				}
				long long int idxSetStartOffset = ((long long int) outerLoopIdx + (long long int) gpuIdx) * (long long int) RUN_MAX_BLK * (long long int) MAX_DEPTH;

				uint* IAPAddr = idxSet + idxSetStartOffset;
				cudaStatus = cudaMemcpy( DM[gpuIdx].firstIdxSet,
									     IAPAddr,
									     sizeof(uint) * MAX_DEPTH * RUN_MAX_BLK,
									     cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in cudaMemcpy for firstIdxSet of the GPU device["<< gpuIdx << "]" << endl;
				}
			}

			if(inOuterLoop == false) {
				break;
			}

			// process an inner join operand
			for(uint pageIdx = 0; pageIdx < numOfPages; pageIdx++)
			{
				// replicating an inner join operand
				for (uint gpuIdx = 0; gpuIdx < numOfGPUs; gpuIdx++)
				{
					cudaStatus = DM[gpuIdx].setDevice(gpuIdx);
					if (cudaStatus != cudaSuccess) {
						cerr 	<< "supportCountingIndexDeliveryChangeOperands():: Error in setting the GPU device["	<< gpuIdx << "]" << endl;
					}
					uint streamOffset = pageIdx % numOfStreams;

					if (isMaterialization == true) {

						uint numOfpartialBitmap = numOfFrequentItemsetsPreviousLevel - (materialization->rb_mgr->numOfItemsetsMaterialized * firstActiveBlock );

						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
	     											    sourceVector[pageIdx]->dataForPage + (firstActiveBlock * verticalListLength * materialization->rb_mgr->numOfItemsetsMaterialized),
                                      				    sizeof(uint) * (numOfpartialBitmap) * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}
						uint batch = (RUN_MAX_BLK - 1) / MAX_GPU_BLOCK + 1;
						for(uint loopIdx = 0; loopIdx < batch ; loopIdx++) {
							Kernel_HFI<<< MAX_GPU_BLOCK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
							( DM[gpuIdx].MiningBuffers[streamOffset].resultVerticalList,
							  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
							  DM[gpuIdx].firstIdxSet,
							  numOfCandidates,
							  lenOfList,
							  depthMax,
							  ((outerLoopIdx+gpuIdx) * RUN_MAX_BLK) + (loopIdx * MAX_GPU_BLOCK),
							  RUN_MAX_BLK
							);
						}

					}
					else {
						cudaStatus = cudaMemcpyAsync( 	DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
	     											    sourceVector[pageIdx]->dataForPage,
	     											    sizeof(uint) * numOfFrequentItemsetsPreviousLevel * lenOfList,
	     											    cudaMemcpyHostToDevice,
	     											    DM[gpuIdx].mPageStreams[streamOffset]);

						if (cudaStatus != cudaSuccess) {
							cerr 	<< "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for sourceVerticalList" << endl;
						}


						uint batch = (RUN_MAX_BLK - 1) / MAX_GPU_BLOCK + 1;
						for(uint loopIdx = 0; loopIdx < batch ; loopIdx++) {
							Kernel_TFL<<<MAX_GPU_BLOCK, MAX_THREAD, 0, DM[gpuIdx].mPageStreams[streamOffset]>>>
							( DM[gpuIdx].MiningBuffers[streamOffset].sourceVerticalList,
						      DM[gpuIdx].MiningBuffers[streamOffset].supportList,
							  DM[gpuIdx].firstIdxSet,
							  numOfCandidates,
							  lenOfList,
							  depth,
							  depthMax,
							  ((outerLoopIdx+gpuIdx) * RUN_MAX_BLK) + (loopIdx * MAX_GPU_BLOCK),
 							  RUN_MAX_BLK
 							);

						}
					}
					// copying the supports of candidate itemsets to the main memory
					uint* partialSupportArr = supportVector[pageIdx]->supportArr + (outerLoopIdx + gpuIdx) * RUN_MAX_BLK;
					cudaStatus = cudaMemcpyAsync( partialSupportArr,
												  DM[gpuIdx].MiningBuffers[streamOffset].supportList,
												  sizeof(uint) * RUN_MAX_BLK,
												  cudaMemcpyDeviceToHost,
												  DM[gpuIdx].mPageStreams[streamOffset]);
					 if (cudaStatus != cudaSuccess) {
						 cerr << "supportCountingIndexDelivery():: Error in cudaMemcpyAsync for supportArr" << endl;
					 }
				}

			}

		}

		 // Synchronizing all the GPUs
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
		// Aggregating partial supports 
		 for (int pageIdx = 0; pageIdx < numOfPages; pageIdx++) {
			 for (int candidateIdx = 0; candidateIdx < numOfCandidates; candidateIdx++) {
				 support[candidateIdx] += supportVector[pageIdx]->supportArr[candidateIdx];
			 }
		 }
	}

	// Release the buffers on the main memory
	void destroyFramework() {
		saVec->destroySAUnitVector();
		cudaError_t cudaStatus;
		if (wholeVerticalList) {
			for (int idx = 0; idx < numOfPages; idx++) {
				delete[] wholeVerticalList[idx];
				wholeVerticalList[idx] = NULL;
			}
			delete[] wholeVerticalList;
			wholeVerticalList = NULL;
		}
		if (idxSetArray) {
			delete[] idxSetArray;
		}

		// the original TBP
		if (initialPageset) {
			initialPageset->destroyPageset();
			delete initialPageset;
			initialPageset = NULL;
		}
		if (supportArray) {
			//delete[] supportArray;
			cudaStatus = cudaFreeHost(supportArray);
			if (cudaStatus != cudaSuccess) {
				cerr
						<< "destroyFramework():: Error in cudaFreeHost for supportArray"
						<< endl;
			}
			supportArray = NULL;
		}

		if (isMaterialization == true) {
			materialization->destroyMaterialization(DM);
		}
		if (DM) {
			for (uint loopIdx = 0; loopIdx < numOfGPUs; loopIdx++) {
				this->DM[loopIdx].setDevice(loopIdx);
				this->DM[loopIdx].destroyStreams();
				this->DM[loopIdx].cudaFreeMiningBuffers();

			}
			delete[] this->DM;
			this->DM = NULL;

		}
		if (candidate_list) {
			for (int depthIdx = 0; depthIdx < maxDepth; depthIdx++) {
				delete[] candidate_list[depthIdx];
				candidate_list[depthIdx] = NULL;
			}
			delete[] candidate_list;
			candidate_list = NULL;
		}
		if (validate) {
			delete[] validate;
			validate = NULL;
		}
		if (p_boundary_list) {
			delete p_boundary_list;
			p_boundary_list = NULL;
		}
		if (new_candidate_list) {
			for (int depthIdx = 0; depthIdx < maxDepth; depthIdx++) {
				delete[] new_candidate_list[depthIdx];
				new_candidate_list[depthIdx] = NULL;
			}
			delete[] new_candidate_list;
			new_candidate_list = NULL;
		}
		if (new_p_boundary_list) {
			delete new_p_boundary_list;
			new_p_boundary_list = NULL;
		}
		if (cntArr) {
			delete cntArr;
			cntArr = NULL;
		}
	}

public:
// the variables on the main memory
	PagesetTmp* initialPageset;

	vector<vector<unsigned> > frequentItemsets;

	uint** wholeVerticalList;

	// the materialization for fast support counting
	Materialization* materialization;

	bool isMaterialization;

	SAUnitVector* saVec;


	int numOfTransactions;
	int numOfPages;
	int numOfStreams;
	int numOfGPUs;

	int numOfCandidates;
	int verticalListLength;
	int wholeVerticalListLength;
	int minsupAbsoulte;

	int level;
	int numOfSingletons;

	int numOfPrefixes;
	uint* idxSetArray;
	uint* supportArray;
	int maxDepth;

	map<int, int> mapFrequentSingletons;

	// the variables on the device memory
	//buffer on the device memory
	DeviceMemoryForStream* DM;

	map<uint, uint> mapSingletonIndex;

	// the variables from GPApriori to generate candidate itemsets using the trie data structure
	unsigned short** candidate_list;
	unsigned short* validate;
	vector<int>* p_boundary_list;
	int candidate_list_length;
	int candidate_list_width;

	// the variables for the pingpong execution
	unsigned short** new_candidate_list;
	vector<int>* new_p_boundary_list;

	// the variables for the parallelization of recordIndexes()
	int numOfCPUs;
	int* cntArr;

	vector<vector<uint> > positionVector;

	uint minSingleton;
	bool isOutput;
	ofstream fout;

	void cleanWholeVerticalList() {
		if (wholeVerticalList) {
			for (int idx = 0; idx < numOfCandidates; idx++) {
				delete[] wholeVerticalList[idx];
				wholeVerticalList[idx] = NULL;
			}
			delete[] wholeVerticalList;
			wholeVerticalList = NULL;
		}
	}

	// returning true if two itemsets are joinable
	bool joinable(vector<unsigned> left, vector<unsigned> right) {
		int vecSiz = left.size() - 1;
		for (int vecIdx = 0; vecIdx < vecSiz; vecIdx++) {
			if (left[vecIdx] != right[vecIdx]) {
				return false;
			}
		}
		return true;
	}
	// pruning infrequent itemsets among candidate itemsets checked in the current iteration
	int parallelPruningCandidates(unsigned short** _candidate_list) {
		int i, j;

		int candidate_itemset_cnt = 0;


		int block_length = (candidate_list_length) / CPU_THREAD;
		int nthreads, tid;
		int startCandidateIdx, endCandidateIdx;
		memset(cntArr, 0, sizeof(int) * (CPU_THREAD));
		omp_set_num_threads(CPU_THREAD);
#pragma omp parallel private(i, nthreads, tid, startCandidateIdx, endCandidateIdx)
		{
			tid = omp_get_thread_num();
			startCandidateIdx = tid * block_length;
			if (tid == CPU_THREAD - 1) {
				endCandidateIdx = candidate_list_length;
			} else {
				endCandidateIdx = (tid + 1) * block_length;
			}

			for (i = startCandidateIdx; i < endCandidateIdx; i++) {
				if (supportArray[i] >= minsupAbsoulte) {
					validate[i] = 1;
					cntArr[tid]++;
				} else {
					validate[i] = 0;
				}
			}
		}
		for (int idx = 0; idx < CPU_THREAD; idx++) {
			candidate_itemset_cnt += cntArr[idx];
		}

		return candidate_itemset_cnt;

	}
	int pruningCandidates(uint** _candidate_list) {
		int i, j;
		int cnt = 0;

		cout << "pruningCandidates():: minsupAbsoulte - " << minsupAbsoulte
				<< endl;
		for (i = 0; i < candidate_list_length; i++) {
			if (supportArray[i] >= minsupAbsoulte) {
				validate[i] = 1;
				cnt++;

			} else {
				validate[i] = 0;
			}
		}
		return cnt;

	}

	void candidateGenerationTrie( unsigned short** _candidate_list,
								  unsigned short** _new_candidate_list,
								  vector<int>* _p_boundary_list,
								  vector<int>* _p_new_boundary_list,
								  unsigned short* _validate,
								  uint* _idxSetFirstSource) {
		cout <<"Generating candidate itemsets. [";
		Timer start, end;
		start = getCurrentTime();
		minSingleton = 65535;
		int boundary_list_length = _p_boundary_list->size();
		int new_candidate_list_length = 0;
		int new_candidate_list_width = 0;
		_p_new_boundary_list->clear();
		set<uint> ck_singleton_per_iteration;
		int i, j = 0, k, u, v;

		// finding the boundaries which used to classify candidate itemsets having the different prefixes
		for (i = 0; i < boundary_list_length; i++) {
			for (; j < (*_p_boundary_list)[i]; j++) {
				if (_validate[j] == 0) {
					continue;
				}
				for (k = j + 1; k < (*_p_boundary_list)[i]; k++) {
					if (_validate[k] == 0) {
						continue;
					}
					new_candidate_list_length++;
				}
			}
		}
		// increasing the widths of candidate itemsets
		new_candidate_list_width = candidate_list_width + 1;

		// clearing the buffer with respect to candidate itemsets
		for (i = 0; i < new_candidate_list_width; i++) {
			memset((void*) _new_candidate_list[i], 0, sizeof(unsigned short) * new_candidate_list_length);
		}
		j = 0;
		v = 0;

		// generating candidate itemsets using tries
		for (i = 0; i < boundary_list_length; i++) {
			for (; j < (*_p_boundary_list)[i]; j++) {
				if (_validate[j] == 0 || j == (*_p_boundary_list)[i] - 1) {
					continue;
				}
				for (k = j + 1; k < (*_p_boundary_list)[i]; k++) {
					if (_validate[k] == 0) {
						continue;
					}

					for (u = 0; u < candidate_list_width; u++) {
						_new_candidate_list[u][v] = _candidate_list[u][j];
						if(minSingleton > _candidate_list[u][j]) {
							minSingleton = _candidate_list[u][j];
						}
					}
					_new_candidate_list[u][v] = _candidate_list[u - 1][k];
					if(minSingleton > _candidate_list[u - 1][k]) {
						minSingleton = _candidate_list[u - 1][k];
					}
					v++;
				}
				_p_new_boundary_list->push_back(v);
			}
		}

		candidate_list_length = new_candidate_list_length;
		candidate_list_width = new_candidate_list_width;
		memset((void *) _validate, 1, (sizeof(unsigned short) * candidate_list_length));
		end = getCurrentTime();

		// recording the position sets of candidate itemsetss generated in the current iteration
		if (isMaterialization == false) {
			int srcIdx = 0;
			long long int srcIdxStartOffset = 0;
			for(int candidateIdx = 0; candidateIdx < candidate_list_length; candidateIdx++) {
				srcIdxStartOffset = (long long int)srcIdx * (long long int)maxDepth;
				srcIdx++;
				for(int lenIdx = 0; lenIdx < candidate_list_width; lenIdx++) {
					idxSetArray[srcIdxStartOffset++] = (uint)_new_candidate_list[lenIdx][candidateIdx];
				}
			}
		}
		cout <<(end-start)<<" sec]"<<endl;
	}


	void parallelRecordIndexes_fixed_width() {

 		map<uint, uint>::iterator blkIterator = materialization->rb_mgr->mapSingletonToBlkID.find(minSingleton);
		uint firstActiveBlock = blkIterator->second;//this->minSingleton;
		int nthreads, tid; /* Variables for indexing thread IDs of OpenMP*/
		int startCandidateIdx, endCandidateIdx;
		long long int srcIdxStartOffset = 0;
		int candidateIdx;
		int lenIdx;

 		map<vector<uint>, uint>::const_iterator materializationIter;
		map<uint, uint>::const_iterator mapSingletonIter;
		unsigned short** _new_candidate_list;
 		int boundaryIdx = 0;
		int boundaryStart = -1;
		int boundaryEnd = -1;
		int cnt = 0;

		if (candidate_list_width % 2 == 1) {
			_new_candidate_list = candidate_list;
		}
		else {
			_new_candidate_list = new_candidate_list;
		}
		
		int block_length = (candidate_list_length) / CPU_THREAD;  

		omp_set_num_threads(CPU_THREAD);
#pragma omp parallel private(nthreads, tid, startCandidateIdx, endCandidateIdx, srcIdxStartOffset, candidateIdx, lenIdx, materializationIter, mapSingletonIter, boundaryIdx, boundaryStart, boundaryEnd, cnt)
		{
			tid = omp_get_thread_num();
			startCandidateIdx = tid * block_length;
			if (tid == CPU_THREAD - 1) {
				endCandidateIdx = candidate_list_length;
			}
			else {
				endCandidateIdx = (tid + 1) * block_length;
			}

			if (isMaterialization == false) {

				for (candidateIdx = startCandidateIdx;	candidateIdx < endCandidateIdx; candidateIdx++) {
					srcIdxStartOffset = (long long int) candidateIdx * (long long int)maxDepth;

					for (lenIdx = 0; lenIdx < candidate_list_width; lenIdx++) {
						mapSingletonIter = this->mapSingletonIndex.find((uint)_new_candidate_list[lenIdx][candidateIdx]);
						idxSetArray[srcIdxStartOffset++] = 	mapSingletonIter->second;//_new_candidate_list[lenIdx][candidateIdx];
					}
				}

			}
			else {
				// for each candidate itemsets, finding their position with pre-computation
				for (candidateIdx = startCandidateIdx;	candidateIdx < endCandidateIdx; candidateIdx++) {
					srcIdxStartOffset = (long long int) candidateIdx * (long long int) maxDepth;
					srcIdxStartOffset++;
					int cnt = 0;
					int prev = -1;
					int next = -1;
					prev = (int)_new_candidate_list[0][candidateIdx] / materialization->fragmentWidth;
					positionVector[tid].clear();
					positionVector[tid].push_back((int)_new_candidate_list[0][candidateIdx] );
					for(lenIdx = 1; lenIdx < candidate_list_width; lenIdx++) {
						next = (int)_new_candidate_list[lenIdx][candidateIdx] / materialization->fragmentWidth;
						// if the 1-itemset is included in the different fragments of the previous 1-itemsets,
						// recording their position 
						if(prev != next) {
							materializationIter = materialization->positionDictionary.find( positionVector[tid]);
							idxSetArray[srcIdxStartOffset++] = materializationIter->second - firstActiveBlock* materialization->rb_mgr->numOfItemsetsMaterialized;
							cnt++;
							positionVector[tid].clear();
							prev = next;
							lenIdx--;
						}
						// else enlarging the current partition 
						else {
							positionVector[tid].push_back((uint)_new_candidate_list[lenIdx][candidateIdx]);
							if (lenIdx == candidate_list_width-1) {
								materializationIter = materialization->positionDictionary.find( positionVector[tid]);
								idxSetArray[srcIdxStartOffset++] = materializationIter->second- firstActiveBlock* materialization->rb_mgr->numOfItemsetsMaterialized;
								cnt++;
							}
						}

					}
					long long int idxSetArrayOffset = (long long int) candidateIdx * (long long int) maxDepth;
					// recording the number of itemset partitions corresponding to the current candidate itemset
					idxSetArray[idxSetArrayOffset] = cnt;
				}
			}
		}

	}

	void ckIndexes() {
		cout << "ckIndexes():: candidate_list_length - "
				<< candidate_list_length << endl;

		for (int candidateIdx = 0; candidateIdx < candidate_list_length;
				candidateIdx++) {
			cout << "candidateIdx - " << candidateIdx << endl;
			for (int lenIdx = 0; lenIdx < maxDepth; lenIdx++) {
				cout << idxSetArray[candidateIdx * maxDepth + lenIdx] << " ";
			}
			cout << endl;
		}
	}
};

#endif /* FRAMEWORK_CUH_ */
