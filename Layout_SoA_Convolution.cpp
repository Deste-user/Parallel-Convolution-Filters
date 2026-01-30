#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct Layout_SoA{
    int * R;
    int * G;
    int * B;
} soa;

soa* load_aos_layout(const cv::Mat &image){
    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;

    soa* soa_data = new soa();
    soa_data->R = new int[N];
    soa_data->G = new int[N];
    soa_data->B = new int[N];

    for (int y=0 ; y< ROWS; y++){
        for (int x=0; x < COLS; x++){
            int idx = y*COLS + x;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y,x);
            soa_data->B[idx]=pixel[0];
            soa_data->G[idx]= pixel[1];
            soa_data->R[idx]= pixel[2];
        }
    }
    return soa_data;
}

cv::Mat load_Mat_from_layout(soa* data,int rows, int cols){
    cv::Mat output =  cv::Mat(rows, cols, CV_8UC3);

    for (int y=0 ; y< rows; y++){
        for (int x=0; x < cols; x++){
            int idx = y * cols + x;
            output.at<cv::Vec3b>(y,x)[0]=data->B[idx];
            output.at<cv::Vec3b>(y,x)[1]=data->G[idx];
            output.at<cv::Vec3b>(y,x)[2]=data->R[idx];
        }
    }
    return output;
}

soa* convolution(const soa* data, const int* kernel, int dim_kernel, int rows, int cols){
    std::cout<<"Convolution with a Kernel of dimention: " << dim_kernel<< "\n"<< std::endl;
    soa* output_data= new soa();
    int N = rows*cols;
    output_data->R = new int[N];
    output_data->G = new int[N];
    output_data->B = new int[N];


    int bound = dim_kernel/2;

    for (int y=0 ; y< rows; y++){
        for (int x=0; x < cols; x++){
            int idx = y * cols + x;
            int sum_R=0;
            int sum_G=0;
            int sum_B=0;

            for (int ky=-bound; ky <= bound; ky++){
                for (int kx=-bound; kx <= bound; kx++){
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    if (nx >= 0 && nx < cols && ny >= 0 && ny < rows ){
                        int i = ny *cols + nx;
                        int k_idx = (ky + bound) * dim_kernel + (kx + bound);
                        int coeff= kernel[k_idx];
                        
                        sum_R += coeff * data->R[i];
                        sum_G += coeff * data->G[i];
                        sum_B += coeff * data->B[i];
                    }
                }
            }

            output_data->R[idx] = std::clamp(sum_R, 0, 255);
            output_data->G[idx] = std::clamp(sum_G, 0, 255);
            output_data->B[idx] = std::clamp(sum_B, 0, 255);

        }
    }
    return output_data;
}

int main(){
    
    cv::Mat image = cv::imread("./img_template.jpeg");
    cv::imshow("Input", image);
    cv::waitKey(0);
    if(image.empty()){
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;

    soa* soa_data = load_aos_layout(image);

    const int filter [9]= {
    0,-1,0,
    -1,5,-1,
    0,-1,0   
    };

    soa* output_data = convolution(soa_data,filter, 3,ROWS,COLS);


    cv::imshow("Output",load_Mat_from_layout(output_data, ROWS,COLS));
    cv::waitKey(0);
    return 0;
}
