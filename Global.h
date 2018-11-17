/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <iostream>
#include <string>
#include <sys/time.h>
#include <map>
#include <boost/unordered_map.hpp>
//#include <boost/unordered_map.hpp>
#include <sstream>
#include <iomanip>
#include <vector>


#include "item.h"

using namespace std;

const uint unit_num_candidate = 20;
const uint unit_num_thread = 64;

// a parameter for materializations
int NUM_HOT_ITEMS = 50;

// configuration factor about an itemset
int MAX_NUM_CANDIDATE = 100000000;
int MAX_LEN_ITEMSET = 128;
const int MAX_DEPTH = 25;
#define MAX_SHARE_CANDIDATE 256 //16

// parameters for the kernel function
#define MAX_ITEMSET_PABUF 20000000//10000000//1000000//16384//10000000//524288//10000000//524288//131072//1048576 // 8388608//131072 //16384
#define MIN_BLOCK 4096
#define MAX_THREAD 64
#define MAX_BLOCK_MAT 16384


#define MAX_GPU_BLOCK 16384 //32768//16384//32768//16384//8192//32768//16384//8192//16384//262144//1000000//262144//16384//262144//16384


#define MAX_NUM_PRECOMPUTATION 10000
uint RUN_MAX_BLK=0;
// parameter for the CPU runtime
#define CPU_THREAD 128
int ckErr =0;

typedef double Timer;


typedef map<vector<uint>, uint> idxType;

bool opMemManager = true;

// elapsed time for candidate generation over all iterations
double elapsedTimeCandidateGeneration;
// elapsed time for support counting over all iterations
double elapsedTimeBuildMaterialization;

// input parameters of run time
struct inputParameter {
	inputParameter() {
		inputPath = "webdocs";
		outputPath = "default_output";
		minsup = 0.1;
		verticalListLength = 4096;
		numOfGPUs = 1;
		numOfStreams = 1;
		isMaterialization = 0;
		sizeOfFragments = 3;
		isWriteOutput = 0;
	}
	void printInputParameter() {
		cout <<"input - " << inputPath <<", output - " << outputPath <<", minsup - "<< minsup<<", verticalListLength - " << verticalListLength <<endl;
		cout <<"numOfGPUs - " << numOfGPUs <<", numOfStreams - " << numOfStreams <<", isMat - "<< isMaterialization <<", sizeofFragments - " << sizeOfFragments <<", isWirteOutput - " << isWriteOutput << endl;
	}
	char* inputPath;
	char* outputPath;
	float minsup;
	int verticalListLength;
	int numOfGPUs;
	int numOfStreams;
	int isMaterialization;
	int sizeOfFragments;
	int isWriteOutput;
};
void print_vector(vector<int> v)
{
	for(int i = 0; i < v.size(); i++)
	{
		cout << v[i]<<" ";
	}
	cout << endl;
}

// converting a value of integer to a value of string
std::string IntToString ( int number )
{
  std::ostringstream oss;

  // Works just like cout
  oss<< number;

  // Return the underlying string
  return oss.str();
}

// triming a variable of string (separated by " ")
std::string TrimRight(const std::string& s)
{
    if (s.length() == 0)
        return s;

    int e = s.find_last_not_of(" ");

    if (e == string::npos)
        return "";

    return std::string(s, 0, e + 1);
}



// setting bits to verticalList
// this function is gotten from FRONTIER_EXPANSION
void setbit(uint* verticalList,  int index, int data) {
	int seg = index / (sizeof(unsigned int) * 8);
	int offset = index % (sizeof(unsigned int) * 8);
	uint bit_mask = 0x80000000;
	bit_mask = bit_mask >> offset;
	if (data == 1) {
		verticalList[seg] = verticalList[seg] | bit_mask;
	}
	else if (data == 0)	{
		verticalList[seg] = verticalList[seg] & (~bit_mask);
	}
}
// getting bits from bit vectors
// this function is gotten from FRONTIER_EXPANSION
int getbit(unsigned int* verticalList, int index) {
	int seg = index / (sizeof(unsigned int) * 8);

	int offset = index % (sizeof(unsigned int) * 8);

	unsigned int bit_mask = 0x80000000;

	bit_mask = bit_mask >> offset;

	if ((verticalList[seg] & bit_mask) == 0) {
		return 0;
	}
	else {
		return 1;
	}
}


int bitcnt(unsigned int src) {
	src = (src & 0x55555555) + ((src >> 1) & 0x55555555);
	src = (src & 0x33333333) + ((src >> 2) & 0x33333333);
	src = (src & 0x0f0f0f0f) + ((src >> 4) & 0x0f0f0f0f);
	src = (src & 0x00ff00ff) + ((src >> 8) & 0x00ff00ff);
	src = (src & 0x0000ffff) + ((src >> 16) & 0x0000ffff);
	return src;
}




// returning the string after reversing it
// it is usually used to check the bitset
string reverse(string to_reverse) {
	string result;
	for(int i=to_reverse.length()-1; i>=0;i--) {
		result+=to_reverse[i];
	}
	return result;
}

// transforming an integer to binary
// it is usually used to check the bitset
string int_to_binary(uint number) {
	static string result;
	static int level = 0;
	level++;
	if(number > 0) {
		if (number%2 == 0) {
			result.append("0");
		}
		else {
			result.append("1");
		}
		int_to_binary(number/2);
	}
	if(level == 1) {
		return reverse(result);
	}
	else {
		return result;
	}
}

// transforming an integer to binary (type of char*)
// it is usually used to check the
char* decimal_to_binary(uint dec) {
	char* des = new char[32];
	//memset(des, '0', 32);
	for (int idx =0 ;idx < 32;idx++) {
		des[idx]='0';
	}
	for (int pos = 31; pos >= 0; --pos) {
		if(dec % 2) {
			des[pos] = '1';
		}
		dec /=2;
	}
	return des;
}


// getting current time
double getCurrentTime() {
	struct timeval curr;
	struct timezone tz;
	gettimeofday(&curr, &tz);
	double tmp = static_cast<double>(curr.tv_sec) * static_cast<double>(1000000) + static_cast<double>(curr.tv_usec);
	return tmp * 1e-6;
}

#endif /* GLOBAL_H_ */
