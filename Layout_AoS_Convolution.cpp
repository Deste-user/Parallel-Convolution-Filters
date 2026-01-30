#include <iostream>
#include <opencv2/opencv.hpp>


typedef struct AoS_Layout{
    unsigned char R;
    unsigned char G;
    unsigned char B;
} aos;

aos* load_aos_layout(const cv::Mat &image){
    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;
    aos *aos_data = new aos[N];

    for (int y=0 ; y< ROWS; y++){
        for (int x=0; x < COLS; x++){
            int idx = y*COLS + x;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y,x);
            aos_data[idx].B=pixel[0];
            aos_data[idx].G= pixel[1];
            aos_data[idx].R= pixel[2];
        }
    }
    return aos_data;
}

cv::Mat load_Mat_from_layout(aos* data,int rows, int cols){
    cv::Mat output =  cv::Mat(rows, cols, CV_8UC3);

    for (int y=0 ; y< rows; y++){
        for (int x=0; x < cols; x++){
            int idx = y * cols + x;
            output.at<cv::Vec3b>(y,x)[0]=data[idx].B;
            output.at<cv::Vec3b>(y,x)[1]=data[idx].G;
            output.at<cv::Vec3b>(y,x)[2]=data[idx].R;
        }
    }
    return output;
}

aos* convolution(const aos* data, const int* kernel, int dim_kernel, int rows, int cols){
    std::cout<<"Convolution with a Kernel of dimention: " << dim_kernel<< "\n"<< std::endl;
    aos* output_data= new aos[rows*cols];

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
                        
                        sum_R += coeff * data[i].R;
                        sum_G += coeff * data[i].G;
                        sum_B += coeff * data[i].B;
                    }
                }
            }

            output_data[idx].R = std::clamp(sum_R, 0, 255);
            output_data[idx].G = std::clamp(sum_G, 0, 255);
            output_data[idx].B = std::clamp(sum_B, 0, 255);

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

    aos* aos_data = load_aos_layout(image);

    const int filter [9]= {
    0,-1,0,
    -1,5,-1,
    0,-1,0   
    };

    aos* output_data = convolution(aos_data,filter, 3,ROWS,COLS);


    cv::imshow("Output",load_Mat_from_layout(output_data, ROWS,COLS));
    cv::waitKey(0);
    return 0;
}
