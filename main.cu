#include <fstream>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#define NUM 10000
#define BLOCK_W 16
#define BLOCK_H 16
#define RADIUS 1 // The radius of a filter 3x3 is one.
#define SMEM_W (BLOCK_W + 2*RADIUS)
#define SMEM_H (BLOCK_H + 2*RADIUS)
#define MAX_FILTER 900


#define IMAGE_TEMPLATE_PATH "/home/deste00/Convolution_Filter/img_template.jpeg"
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
// 108 byte allocated on the constant memory
// The constant memory has a size of 64 Kb so is ok to load this vectors.
__constant__ float d_filter [MAX_FILTER];


// we use # to trasform the code in text. In this case the statement code is trasformed in success or not.
#define CUDA_CHECK_RETURN(value) CheckCUDAErrorAux(__FILE__,__LINE__, #value, value)

typedef struct AoS_Structure{
    uchar R;
    uchar G;
    uchar B;
} AoS;

typedef struct SoA_Structure{
    uchar* R = nullptr;
    uchar* G = nullptr;
    uchar* B = nullptr;
} SoA;


static void CheckCUDAErrorAux(const char* file,unsigned line ,const char* statement,cudaError_t err){
    if (err == cudaSuccess){
        return;
    }else{
        std::cerr << statement << "returned" << cudaGetErrorString(err) << '('<< err << ") at " << file << "and line:" << line << "\n"<< std::endl;
    }
    exit(1);
}


SoA elaborate_img_load_soa(const std::string path_img, int* rows, int* cols, const int resize_factor,bool visualization=false){
    cv::Mat image = cv::imread(path_img);
    cv::resize(image, image,cv::Size(), resize_factor, resize_factor, cv::INTER_CUBIC);

    if (visualization){
        cv::imshow("Original Image", image);
    }
    
    *rows= image.rows;
    *cols= image.cols;

    SoA soa_struct;
    soa_struct.R = new uchar[image.rows * image.cols];
    soa_struct.G = new uchar[image.rows * image.cols];
    soa_struct.B = new uchar[image.rows * image.cols];
    
    for (int y = 0; y< image.rows ; y++){
        for (int x = 0 ; x < image.cols; x++){
            int idx = y* image.cols + x;
            //represent a vector of 3 byte
            cv::Vec3b pxl = image.at<cv::Vec3b>(y,x);
            soa_struct.B[idx] = pxl[0];
            soa_struct.G[idx] = pxl[1];
            soa_struct.R[idx] = pxl[2];
        }
    }

    return soa_struct;
}

AoS* elaborate_img_load_aos(const std::string path_img, int* rows, int* cols, const int resize_factor){
    cv::Mat image = cv::imread(path_img);
    cv::resize(image, image,cv::Size(), resize_factor, resize_factor, cv::INTER_CUBIC);
    
    *rows= image.rows;
    *cols= image.cols;

    AoS* aos_struct = new AoS[image.rows * image.cols];
    
    for (int y = 0; y< image.rows ; y++){
        for (int x = 0 ; x < image.cols; x++){
            int idx = y* image.cols + x;
            //represent a vector of 3 byte
            cv::Vec3b pxl = image.at<cv::Vec3b>(y,x);
            aos_struct[idx].B = pxl[0];
            aos_struct[idx].G = pxl[1];
            aos_struct[idx].R = pxl[2];
        }
    }

    return aos_struct;
}

void display_image(AoS* structure, int rows, int cols){
    cv::Mat img=  cv::Mat(rows, cols, CV_8UC3);

    for (int y=0; y < rows; y++){
        for (int x=0; x < cols; x++){
            int idx = y*cols + x;
            img.at<cv::Vec3b>(y,x)[0]=structure[idx].B;
            img.at<cv::Vec3b>(y,x)[1]=structure[idx].G;
            img.at<cv::Vec3b>(y,x)[2]=structure[idx].R;
        }
    }
    cv::imshow("Convolution with AoS Layout", img);
    
}

void display_image(SoA structure, int rows, int cols){
    cv::Mat img=  cv::Mat(rows, cols, CV_8UC3);

    for (int y=0; y < rows; y++){
        for (int x=0; x < cols; x++){
            int idx = y*cols + x;
            img.at<cv::Vec3b>(y,x)[0]=structure.B[idx];
            img.at<cv::Vec3b>(y,x)[1]=structure.G[idx];
            img.at<cv::Vec3b>(y,x)[2]=structure.R[idx];
        }
    }
    cv::imshow("Convolution with SoA Layout", img);
    
}

//------------------------------- NOT TILING KERNELS--------------------------------------------

__global__ void convolution_SoA(uchar* array_R, uchar* array_G, uchar* array_B, uchar* output, const int height, const int width, const char size){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;

    if (idx_x >= width || idx_y >= height) return;

    char radius = size/2;

    for (int ny=-radius; ny <= radius; ny ++){
        for(int nx=-radius; nx <= radius; nx ++){
            int idx_filter= size*(radius + ny) + (radius + nx);
            float coeff= d_filter[idx_filter];
            
            int cur_x = idx_x + nx;
            int cur_y = idx_y + ny;

            cur_x = max(0, min(cur_x, width - 1));
            cur_y = max(0, min(cur_y, height - 1));

            int true_idx = cur_y *width + cur_x;

            sum_R += array_R[true_idx]*coeff;
            sum_G += array_G[true_idx]*coeff;
            sum_B += array_B[true_idx]*coeff;           

        }
    }
    int N= height*width;
    int idx = idx_y*width + idx_x;
    output[idx] = (uchar)min(max(sum_R, 0.0f), 255.0f);
    output[idx + N] = (uchar)min(max(sum_G, 0.0f), 255.0f);
    output[idx + 2*N] = (uchar)min(max(sum_B, 0.0f), 255.0f);
    return;

}


__global__ void convolution_AoS(const AoS* input_img, AoS* output,const int width,const int height, const char size ){
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;

    if (idx_x >= width || idx_y >= height) return;
    
    char radius = size/2;
   
    for (int ny=-radius; ny <= radius; ny ++){
        for(int nx=-radius; nx <= radius; nx ++){
            int idx_filter = (radius + ny)* size + (radius + nx);
            float coeff = d_filter[idx_filter];

            int cur_x = idx_x + nx;
            int cur_y = idx_y + ny;

            cur_x = max(0, min(cur_x, width - 1));
            cur_y = max(0, min(cur_y, height - 1));

            
            int neighbor = cur_y* width + cur_x;

            sum_R += input_img[neighbor].R*coeff;
            sum_G += input_img[neighbor].G*coeff;
            sum_B += input_img[neighbor].B*coeff;
        }
    }

    int idx = idx_y*width + idx_x;
    output[idx].R = (uchar)min(max(sum_R, 0.0f), 255.0f);
    output[idx].G = (uchar)min(max(sum_G, 0.0f), 255.0f);
    output[idx].B = (uchar)min(max(sum_B, 0.0f), 255.0f);
}


//------------------------------- TILING KERNELS--------------------------------------------

__global__ void tiling_convolution_AoS_dynamic(const AoS* input_img, AoS* output,const int width,const int height, const char size){
    char radius =size/2;
    int w_tile= (blockDim.x+2*radius);
    int h_tile= (blockDim.y +2*radius); 
    int tile_size= w_tile*h_tile;

    //Flatten RRRRRRR... | GGGGG.... | BBBB....
    extern __shared__ uchar sh_mem[];
    uchar* R_pointer=&sh_mem[0];
    uchar* G_pointer=&sh_mem[tile_size];
    uchar* B_pointer=&sh_mem[2*tile_size];

    //These are the coordinates of the thread in the block.
    int tx= threadIdx.x;
    int ty = threadIdx.y;

    //These are the coordinates of the pixel out
    int out_x = blockDim.x*blockIdx.x + tx;
    int out_y = blockDim.y*blockIdx.y + ty;

    int src_corner_x = blockDim.x* blockIdx.x - radius;
    int src_corner_y = blockDim.y* blockIdx.y - radius;

    int thread_id = ty*blockDim.x + tx;
    int num_thread = blockDim.x * blockDim.y;
    
    for (int i=thread_id; i< tile_size; i+=num_thread){
        uchar s_x=i%w_tile;
        uchar s_y = i/w_tile;

        int g_y= s_y + src_corner_y;
        int g_x = s_x +src_corner_x;

        g_y = max(0, min(g_y, height - 1));
        g_x = max(0, min(g_x, width - 1));
        int global_idx = g_y*width + g_x;

        R_pointer[i] = input_img[global_idx].R;
        G_pointer[i] = input_img[global_idx].G;
        B_pointer[i] = input_img[global_idx].B;

    }

    __syncthreads();

    if (out_x >= width || out_y >= height) return;

    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;

    int center_x = tx +radius;
    int center_y =ty+ radius;

    for (int ny=-radius;ny<=radius;ny++){
        for (int nx=-radius;nx<=radius;nx++){
            int idx_filter = (radius + ny)* size + (radius + nx);
            float coeff = d_filter[idx_filter];

            int sh_idx = (center_y+ny)*w_tile + (center_x+nx);

            sum_R+=R_pointer[sh_idx]*coeff;
            sum_G+=G_pointer[sh_idx]*coeff;
            sum_B+=B_pointer[sh_idx]*coeff;
        }    
    }

    int idx = out_y*width + out_x;
    output[idx].R = (uchar)min(max(sum_R, 0.0f), 255.0f);
    output[idx].G = (uchar)min(max(sum_G, 0.0f), 255.0f);
    output[idx].B = (uchar)min(max(sum_B, 0.0f), 255.0f);
}

__global__ void tiling_convolution_SoA_dynamic(uchar* array_R, uchar* array_G, uchar* array_B, uchar* output, const int height, const int width, const char size){
    char radius =size/2;
    int w_tile= (blockDim.x+2*radius);
    int h_tile= (blockDim.y +2*radius); 
    int tile_size= w_tile*h_tile;

    //Flatten RRRRRRR... | GGGGG.... | BBBB....
    extern __shared__ uchar sh_mem[];
    uchar* R_pointer=&sh_mem[0];
    uchar* G_pointer=&sh_mem[tile_size];
    uchar* B_pointer=&sh_mem[2*tile_size];

    //These are the coordinates of the thread in the block.
    int tx= threadIdx.x;
    int ty = threadIdx.y;

    //These are the coordinates of the pixel out
    int out_x = blockDim.x*blockIdx.x + tx;
    int out_y = blockDim.y*blockIdx.y + ty;

    int src_corner_x = blockDim.x* blockIdx.x - radius;
    int src_corner_y = blockDim.y* blockIdx.y - radius;

    int thread_id = ty*blockDim.x + tx;
    int num_thread = blockDim.x * blockDim.y;
    
    for (int i=thread_id; i< tile_size; i+=num_thread){
        uchar s_x=i%w_tile;
        uchar s_y = i/w_tile;

        int g_y= s_y + src_corner_y;
        int g_x = s_x +src_corner_x;

        g_y = max(0, min(g_y, height - 1));
        g_x = max(0, min(g_x, width - 1));
        int global_idx = g_y*width + g_x;

        R_pointer[i] = array_R[global_idx];
        G_pointer[i] = array_G[global_idx];
        B_pointer[i] = array_B[global_idx];

    }

    __syncthreads();

    if (out_x >= width || out_y >= height) return;

    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;

    int center_x = tx +radius;
    int center_y = ty+ radius;

    for (int ny=-radius;ny<=radius;ny++){
        for (int nx=-radius;nx<=radius;nx++){
            int idx_filter = (radius + ny)* size + (radius + nx);
            float coeff = d_filter[idx_filter];

            int sh_idx = (center_y+ny)*w_tile + (center_x+nx);

            sum_R+=R_pointer[sh_idx]*coeff;
            sum_G+=G_pointer[sh_idx]*coeff;
            sum_B+=B_pointer[sh_idx]*coeff;
        }    
    }
    int N=width*height;
    int idx = out_y*width + out_x;
    output[idx] = (uchar)min(max(sum_R, 0.0f), 255.0f);
    output[idx + N] = (uchar)min(max(sum_G, 0.0f), 255.0f);
    output[idx + 2*N] = (uchar)min(max(sum_B, 0.0f), 255.0f);
}



std::vector<float> measure_performance(int factor_size, int l,int dimention_block=16, bool tile=false, bool visualization = false){
    int ROWS;
    int COLS;
    int radius= l/2;
    
    std::vector<float> time;

        // Load the images with the correct resize factor
    SoA soa_struct = elaborate_img_load_soa(IMAGE_TEMPLATE_PATH, &ROWS, &COLS, factor_size,visualization);
    AoS* aos_struct = elaborate_img_load_aos(IMAGE_TEMPLATE_PATH, &ROWS, &COLS, factor_size);

    int N = ROWS *COLS;  
    AoS* d_output_aos;
    SoA d_soa_wrapper;
    AoS* d_input_aos;
    AoS* h_output_aos= new AoS[N];
    uchar* d_soa_output;

    int size_in_bytes_aos= (N) * sizeof(AoS);

    uchar* raw_pointer_storage;
    int size_in_bytes_soa = N*3* sizeof(uchar);
    int channel_size = N * sizeof(uchar);

    // In general, the cudaMalloc takes the double pointer and size
    // It takes the address that contains the address of the first array's element.
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_input_aos, size_in_bytes_aos));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_output_aos, size_in_bytes_aos));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &raw_pointer_storage, size_in_bytes_soa));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_soa_output, size_in_bytes_soa));

    d_soa_wrapper.R = raw_pointer_storage;
    d_soa_wrapper.G = raw_pointer_storage + N;
    d_soa_wrapper.B = raw_pointer_storage + 2*N;

    // The Memcpy use only the address of the first element array and the size in byte. 
    // There is also the kind of Copy in this case we have Host to Device. 
    CUDA_CHECK_RETURN(cudaMemcpy(d_input_aos, aos_struct ,size_in_bytes_aos, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_soa_wrapper.R, soa_struct.R, channel_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_soa_wrapper.G, soa_struct.G, channel_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_soa_wrapper.B, soa_struct.B, channel_size, cudaMemcpyHostToDevice));

    // LOADED ALL.
    dim3 blockDim(dimention_block,dimention_block);
    dim3 gridDim((COLS+ blockDim.x -1)/blockDim.x,(ROWS+ blockDim.y - 1)/blockDim.y );
    size_t num_bytes = (dimention_block+ 2*radius)* (dimention_block+ 2*radius)*3*sizeof(unsigned char);

    // To call the kernel defined, in general:
    // 1. the first one is the dimention of the grid -> how many blocks to work with.
    // 2. the second one  is  the dimention of the block -> how many thread for block to work with.
    // The kernel in a grid has the same shared memory

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds= 0.0f;
    int ITERATIONS = 100;

    cudaEventRecord(start);
    for (int i=0; i<ITERATIONS; i++){
        if (tile){
            tiling_convolution_AoS_dynamic<<<gridDim, blockDim, num_bytes>>>(d_input_aos, d_output_aos,COLS, ROWS,l);
        }else{
            convolution_AoS <<<gridDim, blockDim>>>(d_input_aos, d_output_aos,COLS, ROWS, l);
        }
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    time.push_back(milliseconds / ITERATIONS);
    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    milliseconds=0.0f;
    cudaEventRecord(start);
    for(int i=0; i<ITERATIONS; i++){
        if (tile){
            tiling_convolution_SoA_dynamic<<<gridDim, blockDim,num_bytes>>>(d_soa_wrapper.R, d_soa_wrapper.G, d_soa_wrapper.B, d_soa_output, ROWS, COLS, l);
        }else{
            convolution_SoA<<<gridDim, blockDim>>>(d_soa_wrapper.R, d_soa_wrapper.G, d_soa_wrapper.B, d_soa_output, ROWS, COLS, l);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    time.push_back(milliseconds / ITERATIONS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    CUDA_CHECK_RETURN(cudaMemcpy(h_output_aos, d_output_aos ,size_in_bytes_aos, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.R, d_soa_output,channel_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.G, d_soa_output+ N,channel_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.B, d_soa_output + 2*N,channel_size, cudaMemcpyDeviceToHost));


    CUDA_CHECK_RETURN(cudaFree(d_output_aos));
    CUDA_CHECK_RETURN(cudaFree(raw_pointer_storage));
    CUDA_CHECK_RETURN(cudaFree(d_soa_output));
    CUDA_CHECK_RETURN(cudaFree(d_input_aos));

    if (visualization){
    display_image(h_output_aos, ROWS,COLS);
    display_image(soa_struct,ROWS, COLS);
    cv::waitKey(0);
    }
    
    // Cleanup host memory
    delete[] h_output_aos;
    delete[] aos_struct;
    delete[] soa_struct.R;
    delete[] soa_struct.G;
    delete[] soa_struct.B;
    
    return time;
}

__global__ void dummy_warmup_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Fai un po' di matematica inutile per alzare i clock
        float val = 0.0f;
        for (int i = 0; i < 100; i++) {
            val += sinf(idx * i) * cosf(idx);
        }
        data[idx] = val; // Scrittura in memoria (sveglia il controller memoria)
    }
}

void warmup_gpu() {
    cudaFree(0); 

    int size = 1024 * 1024; 
    float* d_dummy;
 
    cudaMalloc(&d_dummy, size * sizeof(float));
    
    // Lancia il kernel un po' di volte per essere sicuri
    for(int i=0; i<5; i++) {
        dummy_warmup_kernel<<<256, 256>>>(d_dummy, size);
    }
    cudaDeviceSynchronize();

    cudaFree(d_dummy);

}



/* Function that create a Gaussian filter */
std::vector<float> create_gaussian_filter (int size, float sigma=-1.0f){
    std::vector<float> filter (size*size);
    int radius = size/2;
    float sum =0.0f;

    if (sigma < 0.0f){
        sigma = radius/2;
    }

    float two_sigma_squared = 2* sigma *sigma;

    for (int ny=-radius; ny <= radius; ny++){
        for (int nx=-radius; nx <= radius; nx++){
            int idx = (radius + ny)* size + (radius + nx);
            float dist_sq =(float) (nx*nx + ny*ny);
            float value = expf(-dist_sq / two_sigma_squared);
            filter[idx] = value;
            sum += value;
        }
    }

    // Normalization
    for (int i =0; i< size*size; i++){
        filter[i] /= sum;
    }

    return filter;
}

// Filter that detect the edges of the image.
std::vector<float> create_edge_detection_filter(int size) {

    std::vector<float> filter(size * size, 0.0f);
    int center = size / 2;
    

    int idx_center = center * size + center;
    
    filter[idx_center - 1] = -1.0f;    
    filter[idx_center + 1] = -1.0f;    
    filter[idx_center - size] = -1.0f; 
    filter[idx_center + size] = -1.0f; 
    filter[idx_center] = 4.0f;         
    
    return filter;
}

// Filter that makes the average of pixel in the neighborhood.
std::vector<float> create_box_filter(int size) {
    std::vector<float> filter(size * size);
    float value = 1.0f / (float)(size * size); 
    
    for(int i=0; i < size*size; i++) {
        filter[i] = value;
    }
    return filter;
}

void allocate_costant_mem(std::vector<float> filter){
    int size = filter.size();
    if (size > MAX_FILTER){
        std::cout << "Error filter to big!"<< std::endl;
        return;
    } 
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_filter, filter.data(),size*sizeof(float)));
}

void reset_costant_memory(){
    // Erase the constant memory for the filter
    float zero_filter[MAX_FILTER] = {0.0f};
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_filter, zero_filter, MAX_FILTER*sizeof(float)));
}


void confront_layout(){
    int size = 3;
    std::vector<float> filter = create_gaussian_filter(size);

    allocate_costant_mem(filter);

    std::vector<int> factor_resize = {1,2,3,4,5,10,15,20,40};
    std::vector<float> time_aos;    
    std::vector<float> time_soa;

    for (auto factor : factor_resize){
        std::vector<float> time = measure_performance(factor, size);
        time_aos.push_back(time[0]);
        time_soa.push_back(time[1]);
    }

    printf("Factor\tAOS time\tSOA time\n");
    for (size_t i = 0; i < factor_resize.size(); i++){
        printf("%d\t%.6f\t%.6f\n", factor_resize[i], time_aos[i], time_soa[i]);
    }

    reset_costant_memory();
    
    // Save in CSV
    std::ofstream csv_file("./experiments_results/performance_layout.csv");
    csv_file << "Factor,AOS time,SOA time\n";
    for (size_t i = 0; i < factor_resize.size(); i++){
        csv_file << factor_resize[i] << "," << time_aos[i] << "," << time_soa[i] << "\n";
    }
    csv_file.close();

}


void tiling_test(){
    std::vector<int> dim_filter = {3, 5, 7, 9, 11, 15, 21, 25};
    int factor_size = 20;
    std::vector<float> soa_tiled_time;
    std::vector<float> soa_notiled_time;
    printf("Size\tSOA Tiled Time\tSOA Notiled Time\n");
    for (auto size : dim_filter){
        std::vector<float> filter = create_gaussian_filter(size);
        allocate_costant_mem(filter);
        //AoS and then SoA, we take the second element.
        std::vector<float> time = measure_performance(factor_size, size,16,true);
        soa_tiled_time.push_back(time[1]);
        time = measure_performance(factor_size, size,16,false);
        soa_notiled_time.push_back(time[1]);
        reset_costant_memory();
        printf("%d\t%.6f\t%.6f\n", size, soa_tiled_time.back(), soa_notiled_time.back());
    }

    // Save in CSV
    std::ofstream csv_file("./experiments_results/performance_tiling.csv");
    csv_file << "Size,SOA Tiled Time,SOA Notiled Time\n";
    for (size_t i = 0; i < dim_filter.size(); i++){
        csv_file << dim_filter[i] << "," << soa_tiled_time[i] << "," << soa_notiled_time[i] << "\n";
    }
    csv_file.close();
}

void visualize_convolution(int size_filter, int factor_size, std::string type_filter){
    std::vector<float> filter;
    if (type_filter == "gaussian") {
        filter = create_gaussian_filter(size_filter);
    } else if (type_filter == "edge") {
        filter = create_edge_detection_filter(size_filter);
    } else if (type_filter == "box") {
        filter = create_box_filter(size_filter);
    }else{
        filter = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
        };
        size_filter = 3;
    }
    allocate_costant_mem(filter);
    measure_performance(factor_size, size_filter ,16,false, true);
    reset_costant_memory();
}


void dimention_block_test(){
    std::vector<int> dim_block = {2, 4, 8, 16, 32};
    int factor_size = 20;
    int size_filter = 11;
    std::vector<float> soa_tiled_time;
    printf("Dim Block\tSOA Tiled Time\n");
    for (auto dim : dim_block){
        std::vector<float> filter = create_gaussian_filter(size_filter);
        allocate_costant_mem(filter);
        std::vector<float> time = measure_performance(factor_size, size_filter, dim,true);
        soa_tiled_time.push_back(time[1]);
        reset_costant_memory();
        printf("%d\t%.6f\n", dim, soa_tiled_time.back());
    }

    // Save in CSV
    std::ofstream csv_file("./experiments_results/performance_dim_block.csv");
    csv_file << "Dim Block,SOA Tiled Time\n";
    for (size_t i = 0; i < dim_block.size(); i++){
        csv_file << dim_block[i] << "," << soa_tiled_time[i] << "\n";
    }
    csv_file.close();
}


int main(int argc, char* argv[]){

    if (argc > 1 && std::string(argv[1]) == "visualize") {
        int size_filter = 11;
        int factor_size = 3;

        std::string type_filter = "";
        if (argc > 2) {
            type_filter = argv[2];
        }
        visualize_convolution(size_filter, factor_size, type_filter);
        return 0;
    }else{

        warmup_gpu();

        //Create dir to save results
        std::filesystem::create_directory("./experiments_results");

        confront_layout();

        tiling_test();

        dimention_block_test();
    }    
     return 0;

}
