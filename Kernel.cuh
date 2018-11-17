/*
 *
 GMiner can find frequent itemsets using computing power of GPUs.

 Copyright (C)2018 Pradyumn Agrawal, Akash Budhauliya, Arjun Gupta

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

// performing parallel prefix sum
__device__ void warpReduce(volatile uint* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void Kernel_Materialization(   uint* source,
										  uint* result,
										  uint* sup_arr,
										  uint* firstIdxSet,
										  uint list_len,
										  uint vlist_len,
										  uint depthMax,
										  uint old_block_pos,
										  uint RUN_MAX_BLK
										  )
{
	__shared__ uint sup[MAX_THREAD];
 	__shared__ uint candidate[MAX_SHARE_CANDIDATE];
	uint iter, i, depthIdx;
	uint tmp;	
	uint bound;
	int current_block_pos = blockIdx.x + old_block_pos;
	uint insert_block_pos = current_block_pos % RUN_MAX_BLK;

	if (current_block_pos >= list_len) {
		return;
	}
	// getting the number of itemset partition
	uint depth = firstIdxSet[depthMax * insert_block_pos];
	// recording the position sets of a candidate itemset
	if (threadIdx.x < depth + 1) {
		candidate[threadIdx.x] = firstIdxSet[insert_block_pos * depthMax + threadIdx.x];
	}
	sup[threadIdx.x] = 0;
	__syncthreads();

	// computing the number of iterations, each of which is performed at a time
	iter = (vlist_len - 1) / blockDim.x + 1;
	for (i = 0; i < iter; i++) {
		int thread_pos = i * blockDim.x + threadIdx.x;
		tmp = 0xffffffff;
		for (depthIdx = 1; depthIdx < depth + 1; depthIdx++) {
			tmp = tmp & source[candidate[depthIdx] * vlist_len + thread_pos];
		}
		result[insert_block_pos * vlist_len + thread_pos] = tmp;
		sup[threadIdx.x] += __popc(tmp);
		__syncthreads();
	}
	// performing parallel prefix sum
	for(bound = blockDim.x / 2; bound > 32; bound >>=1) {
		if (threadIdx.x < bound) {
			sup[threadIdx.x] += sup[threadIdx.x + bound];
		}
		__syncthreads();
	}

		if (threadIdx.x < 32) {
			warpReduce(sup, threadIdx.x);
		}

		if (threadIdx.x == 0) {
			sup_arr[insert_block_pos] = sup[0];
		}
}


__global__ void Kernel_HFI( uint* source,
							uint* result,
							uint* firstIdxSet,
							uint list_len,
							uint vlist_len,
							uint depthMax,
							uint old_block_pos,
							uint MAX_BLK)
{
	__shared__ uint sup[MAX_THREAD];
	__shared__ uint candidate[MAX_SHARE_CANDIDATE];
	uint iter, i, depthIdx;
	uint tmp;	//[MAX_THREAD];
	int current_block_pos = blockIdx.x + old_block_pos;
	uint bound;
	uint insert_block_pos = current_block_pos % MAX_BLK;
	if (current_block_pos >= list_len) {
		return;
	}
	
	// getting the number of itemset partition
	uint depth = firstIdxSet[insert_block_pos * depthMax];
	if (threadIdx.x < depth + 1) {
		candidate[threadIdx.x] = firstIdxSet[insert_block_pos * depthMax + threadIdx.x];
	}
	sup[threadIdx.x] = 0;
	__syncthreads();
	iter = (vlist_len - 1) / blockDim.x + 1;
	for (i = 0; i < iter; i++) {
		int thread_pos = i * blockDim.x + threadIdx.x;
		tmp = 0xffffffff;
		// performing n-way bitwise AND
		for (depthIdx = 1; depthIdx < depth + 1; depthIdx++) {
			tmp = tmp & source[candidate[depthIdx] * vlist_len + thread_pos];
		}

		sup[threadIdx.x] += __popc(tmp);
		__syncthreads();
	}


	// starting parallel prefix sum
	if (threadIdx.x < 32) {
		warpReduce(sup, threadIdx.x);
	}

	if (threadIdx.x == 0) {
		result[insert_block_pos] = sup[0];
	}
}

int W_SZ = 32;

__global__ void Kernel_TFL( uint* source,
		                    uint* result,
		                    uint* firstIdxSet,
		                    uint list_len,
		                    uint vlist_len,
		                    uint depth,
		                    uint depthMax,
		                    uint old_block_pos,
		                    uint MAX_BLK) {

	__shared__ uint sup[MAX_THREAD];
	__shared__ uint candidate[MAX_SHARE_CANDIDATE];

	uint iter, i, depthIdx;
	uint tmp;	//[MAX_THREAD];
	int current_block_pos = blockIdx.x + old_block_pos;
	uint bound;
	if (current_block_pos >= list_len) {
		return;
	}
	uint insert_block_pos = current_block_pos % MAX_BLK;
	// getting the number of itemset partition
	iter = (vlist_len - 1) / blockDim.x + 1;
	if (threadIdx.x < depth) {
		candidate[threadIdx.x] = firstIdxSet[insert_block_pos * depthMax + threadIdx.x];
	}
	sup[threadIdx.x] = 0;
	__syncthreads();

	for (i = 0; i < iter; i++) {

		int thread_pos = i * blockDim.x + threadIdx.x;
		tmp = 0xffffffff;
		// performing n-way bitwise AND
		for (depthIdx = 0; depthIdx < depth; depthIdx++) {
			tmp = tmp & source[candidate[depthIdx] * vlist_len + thread_pos];
		}
		sup[threadIdx.x] += __popc(tmp);
		__syncthreads();
	}

	// starting parallel prefix sum
	for (bound = blockDim.x / 2; bound > 32; bound >>= 1) {
		if (threadIdx.x < bound) {
			sup[threadIdx.x] += sup[threadIdx.x + bound];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(sup, threadIdx.x);
	}
 
	if (threadIdx.x == 0) {
		result[insert_block_pos] = sup[0];
	}
}


#endif /* KERNEL_CUH_ */
