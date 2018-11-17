/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#pragma once
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <sys/sysinfo.h>


#include "Global.h"

#include "data.h"
#include "item.h"
#include "VerticalDatabase.h"

#include "Framework.cuh"
#include "DeviceBuf.cuh"
#include "PageTmp.h"
#include "Materialization.cuh"
#include "BlockPage.h"
#include <cuda.h> // to get memory on the device
#include <cuda_runtime.h> // to get device count


typedef double Timer;

using namespace std;

void printUsage()
{
 	cout <<"GMiner frequent itemset mining implementation"<<endl;
	cout <<"by Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta, 2018"<<endl;

}

bool readInputParameter(int argc, char*argv[], inputParameter* input) {
	printUsage();
	if (argc == 1)
	{
 		cout << "-i <input path (necessary)>" <<endl;
		cout << "-o <output path (in default out)>" <<endl;
		cout << "-s <minimum support (value in range [0.0,1.0], in default 0.1)>"<<endl;
		cout << "-v <length of bit vectors (in four bytes) >" << endl;
		cout << "-g <number of GPUs (in default 1)>" << endl;
		cout << "-a <number of streams (in default 1)>" << endl;
		cout << "-m <pre-computation (0: no, 1: yes, in default 0)>" << endl;
		cout << "-f <size of fragments (in default 3)>" <<endl;
		cout << "-w <write output (0: no, 1: yes, in default 0)>"<<endl;
		return false;
	}
	int argnr = 0;
	while (++argnr < argc)
	{
		if (!strcmp(argv[argnr], "-i")) {
			input->inputPath = argv[++argnr];
		}
		else if (!strcmp(argv[argnr], "-o")){
			input->outputPath = argv[++argnr];
		}
		else if (!strcmp(argv[argnr], "-s")){
			input->minsup = atof(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-v")){
			input->verticalListLength = atoi(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-g")){
			input->numOfGPUs = atoi(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-a")){
			input->numOfStreams = atoi(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-m")){
			input->isMaterialization = atoi(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-f")){
			input->sizeOfFragments = atoi(argv[++argnr]);
		}
		else if (!strcmp(argv[argnr], "-w")){
			input->isWriteOutput = atoi(argv[++argnr]);
		}


	}
	input->printInputParameter();
	return true;

}
int main(int argc, char *argv[])
{
 	inputParameter* param = new inputParameter();
	Timer startBuildingVerticalDB, endBuildingVerticalDB;
	Timer startBuildingMiningFramework, endBuildingMiningFramework;
	Timer startMining, endMining;

	if (readInputParameter(argc, argv, param) == false) {
		return -1;
	}
	int verticalLength;
	int numOfGPUs = param->numOfGPUs;
	int numOfStreams = param->numOfStreams;
	int sizeOfWidths = param->sizeOfFragments;
	int isMat = param->isMaterialization;
	int isOutput = param->isWriteOutput;
	verticalDatabase vd;
	vd.setData(param->inputPath);
	vd.setMinsupRelative(param->minsup);
	verticalLength = param->verticalListLength;
	startBuildingVerticalDB = getCurrentTime();
	vd.buildVerticalDatabase(param->sizeOfFragments);
	endBuildingVerticalDB = getCurrentTime();
	Framework framework;
	elapsedTimeCandidateGeneration = 0;
	startBuildingMiningFramework = getCurrentTime();

	framework.initFramework(vd, param, MAX_DEPTH);

	endBuildingMiningFramework = getCurrentTime();
	startMining = getCurrentTime();
	double startLevel, endLevel;
	int iteration=0;
	startLevel = getCurrentTime();


	int returnNextLevel = framework.nextLevel();
 	endLevel = getCurrentTime();

	cout <<"Performing Level " << iteration + 2 << ". [";
	cout<<(endLevel - startLevel)<<" sec]"<<endl;
	cout <<"- the # of frequent itemsets - " << returnNextLevel << endl;
	iteration++;

	while(returnNextLevel >= 1) {
		startLevel = getCurrentTime();
		returnNextLevel =  framework.nextLevel();
		endLevel = getCurrentTime();
		cout <<"Performing Level " << iteration++ + 2 << ". [";
		cout<<(endLevel - startLevel)<<" sec]"<<endl;
		cout <<"- the # of frequent itemsets - " << returnNextLevel << endl;

  	}
	endMining = getCurrentTime();
	cout <<"Presenting the information of runtime"<<endl;
	cout <<"- dataset: " << param->inputPath <<", minsup: " << param->minsup << endl;
	cout <<"- vertical list length: " << param->verticalListLength<< endl;
	cout <<"- the # of streams: " << param->numOfStreams << endl;
	cout <<"- the maximum GPU blocks: " << MAX_ITEMSET_PABUF << endl;
	cout <<"- the maximum GPU threads per block: " << MAX_THREAD << endl;
	cout <<"- the # of GPUs: " << numOfGPUs << endl;
	if(isMat == 1) {
	cout <<"- the size of fragments: " << sizeOfWidths << endl;
	}

	cout <<"Presenting the elapsed time for each steps" << endl;
	cout <<"Building vertical database. [" << (endBuildingVerticalDB - startBuildingVerticalDB)<<" sec]"<<endl;
	cout <<"Building mining framework. [" << (endBuildingMiningFramework - startBuildingMiningFramework)<<" sec]"<<endl;
	cout <<"Generating pre-computation. [" << (elapsedTimeBuildMaterialization) << " sec]" << endl;
	cout <<"Performing mining operations. [" << (endMining - startMining)<<" sec]"<<endl;
	cout <<"- Generating candidate itemsets. [" <<elapsedTimeCandidateGeneration<<" sec]"<<endl;
	cout <<"- Performing support counting. [" << (endMining-startMining-elapsedTimeCandidateGeneration)<<" sec]"<<endl;

	framework.destroyFramework();
	if(param) {
		delete param;
	}
}
 
