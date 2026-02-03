#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#define NUM 10000
#define IMAGE_TEMPLATE_PATH "/home/deste00/Convolution_Filter/img_template.jpeg"
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
// 108 byte allocated on the constant memory
// The constant memory has a size of 64 Kb so is ok to load this vectors.
__constant__ float d_filter [9];


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


SoA elaborate_img_load_soa(const std::string path_img, int* rows, int* cols, const int resize_factor){
    cv::Mat image = cv::imread(path_img);
    cv::resize(image, image,cv::Size(), resize_factor, resize_factor, cv::INTER_CUBIC);
    
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


int main(int argc, char** argv){
    int ROWS;
    int COLS;
    // Load the images
    SoA soa_struct = elaborate_img_load_soa(IMAGE_TEMPLATE_PATH, &ROWS, &COLS, 1);
    AoS* aos_struct = elaborate_img_load_aos(IMAGE_TEMPLATE_PATH, &ROWS, &COLS, 1);

    int N = ROWS *COLS;
    std::string type_filter= "sharp";
    if (argc != 1){
    type_filter = argv[1];
    }

    float h_filter [9];

    if (type_filter == "blur") {
        float tmp[9] = {
            1.0f/9, 1.0f/9, 1.0f/9,
            1.0f/9, 1.0f/9, 1.0f/9,
            1.0f/9, 1.0f/9, 1.0f/9
        };
        std::copy(tmp, tmp+9, h_filter);

    } else if (type_filter == "gauss") {
        float tmp[9] = {
            1.0f/16, 2.0f/16, 1.0f/16,
            2.0f/16, 4.0f/16, 2.0f/16,
            1.0f/16, 2.0f/16, 1.0f/16
        };
        std::copy(tmp, tmp+9, h_filter);
    }else{
        float tmp[9] = {0,-1,0,-1,5,-1,0,-1,0};
        std::copy(tmp, tmp+9, h_filter);
    }    
   
    cudaMemcpyToSymbol(d_filter,h_filter, 9*sizeof(float));
    int size = ARRAY_SIZE(h_filter);
    int l = sqrt(size);
    
    
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


    //int thread_per_Blocks=256;
    dim3 blockDim(16,16);
    dim3 gridDim((COLS+ blockDim.x -1)/blockDim.x,(ROWS+ blockDim.y - 1)/blockDim.y );
    //int blocks_per_grid = (N + thread_per_Blocks -1)/thread_per_Blocks;

    // To call the kernel defined, in general:
    // 1. the first one is the dimention of the grid -> how many blocks to work with.
    // 2. the second one  is  the dimention of the block -> how many thread for block to work with.
    // The kernel in a grid has the same shared memory
    convolution_AoS <<<gridDim, blockDim>>>(d_input_aos, d_output_aos,COLS, ROWS,l);
    cudaDeviceSynchronize();

    convolution_SoA<<<gridDim, blockDim>>>(d_soa_wrapper.R, d_soa_wrapper.G, d_soa_wrapper.B, d_soa_output, ROWS, COLS, l);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_output_aos, d_output_aos ,size_in_bytes_aos, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.R, d_soa_output,channel_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.G, d_soa_output+ N,channel_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(soa_struct.B, d_soa_output + 2*N,channel_size, cudaMemcpyDeviceToHost));


    CUDA_CHECK_RETURN(cudaFree(d_output_aos));
    CUDA_CHECK_RETURN(cudaFree(raw_pointer_storage));
    CUDA_CHECK_RETURN(cudaFree(d_soa_output));
    CUDA_CHECK_RETURN(cudaFree(d_input_aos));


    display_image(h_output_aos, ROWS,COLS);
    display_image(soa_struct,ROWS, COLS);
    cv::waitKey(0);
    std::cout<< "The end"<< std::endl;

}
