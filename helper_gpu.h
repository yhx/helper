/* This header file is writen by qp09
 * usually just for fun
 * Thu February 16 2017
 */
#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include <assert.h>

#include "helper_cuda.h"

#define GPU_CHECK(val) check ( (val), #val, file, line )

inline void gpuDevice(int device = 0) {
    checkCudaErrors(cudaSetDevice(device));
}

template<typename T>
T* hostMalloc(size_t size = 1)
{
	T * ret;
	checkCudaErrors(cudaMallocHost((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
T* gpuMalloc(size_t size = 1)
{ 
	T * ret;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
void gpuMemset(T* array, int c, size_t size = 1)
{ 
	assert(array);
	checkCudaErrors(cudaMemset(array, 0, sizeof(T)*(size)));
}

template<typename T>
void hostFree(T* cpu)
{
	assert(cpu);
	checkCudaErrors(cudaFreeHost(cpu));
}

template<typename T>
void gpuFree(T* gpu)
{
	assert(gpu);
	checkCudaErrors(cudaFree(gpu));
}

template<typename T>
void gpuMemcpyPeer(T*data_d, int dst, T*data_s, int src, size_t size = 1)
{
	assert(data_d);
	assert(data_s);
	checkCudaErrors(cudaMemcpyPeer(data_d, dst, data_s, src, sizeof(T)*(size)));
}

template<typename T>
void gpuMemcpy(T*data_d, T*data_s, size_t size = 1)
{
	assert(data_d);
	assert(data_s);
	checkCudaErrors(cudaMemcpy(data_d, data_s, sizeof(T)*(size), cudaMemcpyDeviceToDevice));
}

template<typename T>
T* copyToGPU(T* cpu, size_t size = 1)
{
	T * ret;
	assert(cpu);
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));

	return ret;
}

template<typename T>
void copyToGPU(T* gpu, T* cpu, size_t size = 1)
{
	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(gpu, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
}

template<typename T>
T* copyFromGPU(T* gpu, size_t size = 1)
{
	assert(gpu);
	T * ret = static_cast<T*>(malloc(sizeof(T)*size));
	assert(ret);
	checkCudaErrors(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));

	return ret;
}

template<typename T>
void copyFromGPU(T* cpu, T* gpu, size_t size = 1)
{
	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(cpu, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
}


template<typename DATA, typename SIZE>
__device__ int merge2array(DATA *src, const SIZE size, DATA *dst, SIZE * dst_size, const SIZE dst_offset) 
{
	__shared__ volatile SIZE start_loc;
	if (threadIdx.x == 0) {
		start_loc = atomicAdd(dst_size, size);
	}
	__syncthreads();

	for (SIZE idx=threadIdx.x; idx<size; idx+=blockDim.x) {
		dst[dst_offset + start_loc + idx] = src[idx];
	}

	return 0;
}

#endif /* HELPER_GPU_H */

