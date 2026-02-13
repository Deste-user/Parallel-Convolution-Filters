#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#define ITERATIONS 100
// ==================== SEQUENTIAL AoS ==================
typedef struct AoS_Layout {
    unsigned char R;
    unsigned char G;
    unsigned char B;
} AoS;

AoS* load_aos_layout(const cv::Mat &image) {
    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;
    AoS *aos_data = new AoS[N];

    for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < COLS; x++) {
            int idx = y * COLS + x;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            aos_data[idx].B = pixel[0];
            aos_data[idx].G = pixel[1];
            aos_data[idx].R = pixel[2];
        }
    }
    return aos_data;
}

AoS* convolution_aos(const AoS* data, const int* kernel, int dim_kernel, int rows, int cols) {
    AoS* output_data = new AoS[rows * cols];
    int bound = dim_kernel / 2;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            int sum_R = 0;
            int sum_G = 0;
            int sum_B = 0;

            for (int ky = -bound; ky <= bound; ky++) {
                for (int kx = -bound; kx <= bound; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                        int i = ny * cols + nx;
                        int k_idx = (ky + bound) * dim_kernel + (kx + bound);
                        int coeff = kernel[k_idx];

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

// ================= SEQUENTIAL SoA ====================

typedef struct Layout_SoA {
    unsigned char* R;
    unsigned char* G;
    unsigned char* B;
} SoA;

SoA* load_soa_layout(const cv::Mat &image) {
    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;

    SoA* soa_data = new SoA();
    soa_data->R = new unsigned char[N];
    soa_data->G = new unsigned char[N];
    soa_data->B = new unsigned char[N];

    for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < COLS; x++) {
            int idx = y * COLS + x;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            soa_data->B[idx] = pixel[0];
            soa_data->G[idx] = pixel[1];
            soa_data->R[idx] = pixel[2];
        }
    }
    return soa_data;
}

SoA* convolution_soa(const SoA* data, const int* kernel, int dim_kernel, int rows, int cols) {
    SoA* output_data = new SoA();
    int N = rows * cols;
    output_data->R = new unsigned char[N];
    output_data->G = new unsigned char[N];
    output_data->B = new unsigned char[N];

    int bound = dim_kernel / 2;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            int sum_R = 0;
            int sum_G = 0;
            int sum_B = 0;

            for (int ky = -bound; ky <= bound; ky++) {
                for (int kx = -bound; kx <= bound; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                        int i = ny * cols + nx;
                        int k_idx = (ky + bound) * dim_kernel + (kx + bound);
                        int coeff = kernel[k_idx];

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


std::vector<float> create_gaussian_filter(int size, float sigma = -1.0f) {
    std::vector<float> filter(size * size);
    int radius = size / 2;
    float sum = 0.0f;

    if (sigma < 0.0f) {
        sigma = radius / 2.0f;
    }

    float two_sigma_squared = 2 * sigma * sigma;

    for (int ny = -radius; ny <= radius; ny++) {
        for (int nx = -radius; nx <= radius; nx++) {
            int idx = (radius + ny) * size + (radius + nx);
            float dist_sq = (float)(nx * nx + ny * ny);
            float value = expf(-dist_sq / two_sigma_squared);
            filter[idx] = value;
            sum += value;
        }
    }

    // Normalization
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }

    return filter;
}


void dimention_test(const cv::Mat &image) {
    std::vector<int> resize_factors ={1,2,3,4,5,10,20,40};
    std::vector<float> filter = create_gaussian_filter(3);
    std::vector<double> times_soa;
    std::vector<double> times_aos;

    for (auto factor : resize_factors) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(),factor,factor);
        int ROWS = resized.rows;
        int COLS = resized.cols;
        

        AoS* aos_data = load_aos_layout(resized);
        SoA* soa_data = load_soa_layout(resized);
        
        std::cout << "Resized to: " << COLS << "x" << ROWS << std::endl;
        auto start_seq = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            SoA* output = convolution_soa(soa_data, (int*)filter.data(), 3, ROWS, COLS);
            delete[] output->R;
            delete[] output->G;
            delete[] output->B;
            delete output;
        }
        auto end_seq = std::chrono::high_resolution_clock::now();
        double time_seq_soa = std::chrono::duration<double, std::milli>(end_seq - start_seq).count() / ITERATIONS;

        times_soa.push_back(time_seq_soa);

        start_seq = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            AoS* output = convolution_aos(aos_data, (int*)filter.data(), 3, ROWS, COLS);
            delete[] output;
        }    
        end_seq = std::chrono::high_resolution_clock::now();
        double time_seq_aos = std::chrono::duration<double, std::milli>(end_seq - start_seq).count() / ITERATIONS;
        times_aos.push_back(time_seq_aos);
        std::cout << "AoS: " << std::fixed << std::setprecision(3) << time_seq_aos << " ms, "
                  << "SoA: " << std::fixed << std::setprecision(3) << time_seq_soa << " ms\n" << std::endl;
        
        // Cleanup
        delete[] aos_data;
        delete[] soa_data->R;
        delete[] soa_data->G;
        delete[] soa_data->B;
        delete soa_data;
    }
    std::ofstream csv_file("./experiments_results/performance_layout_seq.csv");
    csv_file << "Factor,AOS time,SOA time\n";
    for (size_t i = 0; i < resize_factors.size(); i++){
        csv_file << resize_factors[i] << "," << times_aos[i] << "," << times_soa[i] << "\n";
    }
    csv_file.close();
}


int main() {
    const std::string IMAGE_PATH = "img_template.jpeg";
    
    cv::Mat image = cv::imread(IMAGE_PATH);
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    int ROWS = image.rows;
    int COLS = image.cols;
    int N = ROWS * COLS;

    std::cout << "Original Image Size: " << COLS << "x" << ROWS << std::endl;
    dimention_test(image);

}
