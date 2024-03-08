#include <hip/hip_runtime.h>
#include <iostream>
#define BLOCK_SIZE 512
#define N 4000000

using namespace std;

void CHECK(hipError_t e){
    if(e!=hipSuccess){
        cout<<"Cuda error:"<<hipGetErrorString(e)<<endl;
        exit(0);
    }
}


__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
    unsigned int gridSize = BLOCK_SIZE*2;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+BLOCK_SIZE]; i += gridSize; 
    }
    __syncthreads();
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); 
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); 
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); 
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(){

    // Host data
    int *h_data = new int[N];
    int ans=0;
    for(int i=0; i<N; ++i) {
        h_data[i] = 1;
        ans++;
    }
    
    // Device data
    int *d_data, *d_output;
    hipMallocManaged(&d_data, sizeof(int) * N);
    hipMallocManaged(&d_output, sizeof(int));
    hipMemcpy(d_data, h_data, sizeof(int) * N, hipMemcpyHostToDevice);
    hipDeviceSynchronize();    
    CHECK(hipGetLastError());
    d_output[0] = 0;

    // Kernel
    cout<<"Start the kernel\n";
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid;
    grid.x = (N+BLOCK_SIZE-1) / BLOCK_SIZE;
    grid.x  = 1;
    reduce<<<grid, block, sizeof(int) * BLOCK_SIZE>>>(d_data, d_output, N);
    hipDeviceSynchronize();
    CHECK(hipGetLastError());
    if(d_output[0]==ans)
        cout<<"Success, Ans = "<<d_output[0]<<endl;
    else
        cout<<"Fail. Ans is "<<ans<<", but you got "<<d_output[0]<<endl;
}