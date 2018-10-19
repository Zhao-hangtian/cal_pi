
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

/**
 * Using CPU to calculate pi
 * @param a Lower Integral Bounds
 * @param b Upper Integral Bounds
 * @param Integral Value
 */
const int N = 1024 * 1024 * 64;
void pi_by_cpu(double a, double b, double *integral) {
	int i;
	double x, temp = 0;
	for (i = 0; i < N; i++) {
		x = a + (double)(b - a) / N * (i + 0.5);
		temp += 4 / (1 + x * x);
	}
	temp *= (double)(b - a) / N;
	*integral = temp;
}

// Using CUDA device to calculate pi
#include <stdio.h>
#include <cuda.h>

#define NBIN N  // Number of bins
#define NUM_BLOCK  64  // Number of thread blocks
#define NUM_THREAD  256  // Number of threads per block	
int tid;
double pi = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(double *sum, int nbin, double step, int nthreads, int nblocks) {
	int i;
	double x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;  // Sequential thread index across the blocks
	for (i = idx; i < nbin; i += nthreads * nblocks) {
		x = (i + 0.5)*step;
		sum[idx] += 4.0 / (1.0 + x * x);
	}
}

// Main routine that executes on the host
int main(void) {

	//Using CPU to calculate pi
	double a, b;
	double   integral;
	clock_t  clockBegin, clockEnd;
	double duration;
	a = 0;
	b = 1;
	clockBegin = clock();
	pi_by_cpu(a, b, &integral);
	clockEnd = clock();
	duration = (double)1000 * (clockEnd - clockBegin) / CLOCKS_PER_SEC;
	printf("CPU Result: %.11lf\n", integral);
	printf("CPU Elapsed time: %.6lfms\n\n", duration);

	// Using CUDA device to calculate pi
	cudaEvent_t start, stop;
	cudaEventCreate(&start); //event
	cudaEventCreate(&stop);  //event
	cudaEventRecord(start, 0);  //record time
	float tm;

	dim3 dimGrid(NUM_BLOCK, 1, 1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD, 1, 1);  // Block dimensions
	double *sumHost, *sumDev;  // Pointer to host & device arrays

	double step = 1.0 / NBIN;  // Step size
	size_t size = NUM_BLOCK * NUM_THREAD * sizeof(double);  //Array memory size
	sumHost = (double *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **)&sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	// Do calculation on device
	cal_pi <<<dimGrid, dimBlock >>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for (tid = 0; tid < NUM_THREAD*NUM_BLOCK; tid++)
		pi += sumHost[tid];
	pi *= step;

	// Print results
	printf("GPU Result: %.11lf\n", pi);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tm, start, stop);
	printf("GPU Elapsed time:%.6f ms.\n\n", tm);


	// Cleanup
	free(sumHost);
	cudaFree(sumDev);
	printf("Press to exit.\n");
	getchar();
	return 0;
}

