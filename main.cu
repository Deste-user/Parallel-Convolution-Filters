#include <cuda_runtime.h>
#include <iostream>
//#include <opencv2/opencv.hpp>
#include <vector>
#define NUM 10000
// we use # to trasform the code in text. In this case the statement code is trasformed in success or not.
#define CUDA_CHECK_RETURN(value) CheckCUDAErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCUDAErrorAux(const char* file,unsigned line ,const char* statement,cudaError_t err){
    if (err == cudaSuccess){
        return;
    }else{
        std::cerr << statement << "returned" << cudaGetErrorString(err) << '('<< err << ") at " << file << "and line:" << line << "\n"<< std::endl;
    }
    exit(1);
}

//Called by the Host but executed by the Device
__global__ void Sum_arrays(int* d_element1, int* d_element2, int *output, int N ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = d_element1[idx] + d_element2[idx];
    }
    return;
}


int main(){
    //All pointer contains the address.
    int * h_array1 = new int[NUM];
    int * h_array2 = new int[NUM];
    int * h_output = new int[NUM];
    int * d_output;
    int * d_array1; int * d_array2;
    size_t size_in_bytes = NUM * sizeof(int);
    
    for (int i=0; i< NUM; i++){
        //Address + Offset
        h_array1[i] = i;
        h_array2[i] = i;
    }


    // In general, the cudaMalloc takes the double pointer and size
    // It takes the address that contains the address of the first array's element.
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_output, size_in_bytes));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_array1, size_in_bytes));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_array2, size_in_bytes));


    // The Memcpy use only the address of the first element array and the size in byte. 
    // There is also the kind of Copy in this case we have Host to Device. 
    CUDA_CHECK_RETURN(cudaMemcpy(d_array1, h_array1,size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_array2, h_array2,size_in_bytes, cudaMemcpyHostToDevice));

    int thread_per_Blocks=256;
    int blocks_per_grid = (NUM + thread_per_Blocks -1)/thread_per_Blocks;

    Sum_arrays <<<blocks_per_grid, thread_per_Blocks>>>(d_array1, d_array2, d_output, NUM);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output ,size_in_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_output));
    CUDA_CHECK_RETURN(cudaFree(d_array1));
    CUDA_CHECK_RETURN(cudaFree(d_array2));
    
    for (int i=500;i< 1000 ; i++){
        std::cout<< "Number:"<< h_output[i]<< std::endl; 
    }

    delete[] h_array1;
    delete[] h_array2;
    delete[] h_output;
    return 0;


}
