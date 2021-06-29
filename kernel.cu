
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include<sys/utime.h>
#include <string.h>


using namespace cv;
using namespace std;

/// define gettimeofday on windows
int gettimeofday(struct timeval* tp, void* tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}


double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

#define BLOCK_SIZE 256
#define MAX_THREADS 1024
#define MAX_VALUE INT_MAX

int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}

__global__ void kernelSeamCarving(uchar3* d_inputMat) {
    printf("kernel\n");
    return;
}
int nextPower2(int n) {
    n--;
    n = n >> 1 | n;
    n = n >> 2 | n;
    n = n >> 4 | n;
    n = n >> 8 | n;
    n = n >> 16 | n;
    return ++n;
}

void AddSeamToImgRed(unsigned char* imgPtr, vector<uint> seam, size_t rows, size_t cols, int channels) {
    for (vector<uint>::size_type i = 0; i != seam.size(); i++) {
        imgPtr[i * cols * channels + seam[i] * channels] = 0;
        imgPtr[i * cols * channels + seam[i] * channels + 1] = 0;
        imgPtr[i * cols * channels + seam[i] * channels + 2] = 255;
    }
}
/*
__global__ void computeEnergyGPU(size_t rows, size_t cols, unsigned char* img, unsigned int* energy) {

    extern __shared__ unsigned char tmp[]; // 3 * sizeof(char) * BLOCK_SIZE

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;

    int ny = rows, nx = cols;

    tmp[threadIdx.x * 3] = img[index * 3];
    tmp[threadIdx.x * 3 + 1] = img[index * 3 + 1];
    tmp[threadIdx.x * 3 + 2] = img[index * 3 + 2];

    __syncthreads();

    int ix = index / cols, iy = index % cols;
    if (ix < nx && iy < ny) {
        if (ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1) {
            energy[index] = 20000;
        }
        else {
            int val = 0;
            val += powf(img[(index - nx) * 3] - img[(index + nx) * 3], 2);
            val += powf(img[(index - nx) * 3 + 1] - img[(index + nx) * 3 + 1], 2);
            val += powf(img[(index - nx) * 3 + 2] - img[(index + nx) * 3 + 2], 2);
            if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) {
                val += powf(img[index * 3 + 3] - img[index * 3 - 3], 2);
                val += powf(img[index * 3 + 4] - img[index * 3 - 2], 2);
                val += powf(img[index * 3 + 5] - img[index * 3 - 1], 2);
            }
            else {
                val += powf(tmp[threadIdx.x * 3 + 3] - tmp[threadIdx.x * 3 - 3], 2);
                val += powf(tmp[threadIdx.x * 3 + 4] - tmp[threadIdx.x * 3 - 2], 2);
                val += powf(tmp[threadIdx.x * 3 + 5] - tmp[threadIdx.x * 3 - 1], 2);
            }
            energy[index] = val;
        }
    }
}
*/

__global__ void computeEnergyGPU(size_t rows, size_t cols, unsigned char* img, unsigned int* energy) {
    int  ny = rows;
    int  nx = cols;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = nx * iy + ix;

    if (ix < nx && iy < ny) {
        // set a large number to keep the margin pixels
        if (ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1) {
            energy[idx] = 20000;
        }
        else {
            int val = 0;
            // vertical energy
            val += powf(img[(idx - nx) * 3] - img[(idx + nx) * 3], 2);
            val += powf(img[(idx - nx) * 3 + 1] - img[(idx + nx) * 3 + 1], 2);
            val += powf(img[(idx - nx) * 3 + 2] - img[(idx + nx) * 3 + 2], 2);

            // horizontal energy
            val += powf(img[idx * 3 + 3] - img[idx * 3 - 3], 2);
            val += powf(img[idx * 3 + 4] - img[idx * 3 - 2], 2);
            val += powf(img[idx * 3 + 5] - img[idx * 3 - 1], 2);

            energy[idx] = val;
        }
    }
}

__global__ void compute_min_cost_kernel(unsigned int* energies, unsigned int* min_costs, unsigned int* temp_d, int width, int height, int row) {
    // Extract thread and block index information
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int col = bx * MAX_THREADS + tx;

    if (col >= width)  // for excess threads
        return;

    unsigned int left, right, middle;
    if (bx == 0)
        left = (tx > 0) ? temp_d[tx - 1] : MAX_VALUE;
    else
        left = temp_d[col - 1];
    middle = temp_d[col];
    right = (col < width - 1) ? temp_d[col + 1] : MAX_VALUE;

    unsigned int minimum = min(left, min(middle, right));
    unsigned int cost = minimum + energies[row * width + col];

    __syncthreads();
    temp_d[col] = cost;

    __syncthreads();
    min_costs[row * width + col] = cost;
}

__global__ void find_min_kernel(unsigned int* row, unsigned int* mins, int* min_indices, int width, int power) {
    // Compute current index.
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = tx + bx * power;

    // Set up shared memory for tracking mins.
    extern __shared__ unsigned int shared_memory[];
    unsigned int* shared_mins = (unsigned int*)shared_memory;
    int* shared_min_indices = (int*)(&(shared_memory[power]));

    // Copy global intermediate values into shared memory.
    shared_mins[tx] = (index < width) ? row[index] : MAX_VALUE;
    shared_min_indices[tx] = (index < width) ? index : MAX_VALUE;

    __syncthreads();

    // Do the reduction for value pairs.
    for (int i = power / 2; i > 0; i >>= 1) {
        if (tx < i)
            if (shared_mins[tx] > shared_mins[tx + i])
            {
                shared_mins[tx] = shared_mins[tx + i];
                shared_min_indices[tx] = shared_min_indices[tx + i];
            }
        __syncthreads();
    }

    // Thread 0 has the solution.
    if (tx == 0) {
        mins[bx] = shared_mins[0];
        min_indices[bx] = shared_min_indices[0];
    }
}

__global__ void build_indexMap_kernel(unsigned int* energies, unsigned int* indexMap, unsigned int* offsetMap, int width, int height) {
    int ny = height;
    int nx = width;
    int ix = threadIdx.x + blockIdx.x * blockDim.x;//column
    int iy = threadIdx.y + blockIdx.y * blockDim.y;//row
    unsigned int idx = iy * nx + ix;
    unsigned int left_idx = (iy + 1) * nx + (ix - 1);
    unsigned int mid_idx = (iy + 1) * nx + (ix);
    unsigned int right_idx = (iy + 1) * nx + (ix + 1);

    unsigned int left, right, middle;
    if (ix < nx && iy < ny - 1) {
        if (ix == 0) {
            left = MAX_VALUE;
        }
        else {
            left = energies[left_idx];
        }
        middle = energies[mid_idx];
        if (ix == nx - 1) {
            right = MAX_VALUE;
        }
        else {
            right = energies[right_idx];
        }
        unsigned int minimum = min(left, min(middle, right));
        indexMap[idx] = minimum == left ? left_idx : minimum == right ? right_idx : mid_idx;
        offsetMap[idx] = indexMap[idx];
    }
}

__global__ void build_seamMap_kernel(unsigned int* energies, unsigned int* indexMap, int width, int height, int lineNum) {
    int ny = height;
    int nx = width;
    int ix = threadIdx.x + blockIdx.x * blockDim.x; //column
    int iy = threadIdx.y + blockIdx.y * blockDim.y; //row
    int idx = iy * nx + ix;

    if (ix < nx && iy < ny - 1) {
        if (threadIdx.y == 0) {
            energies[idx] += energies[indexMap[idx]];
            if (indexMap[idx] < (ny - 1) * nx) {
                indexMap[idx] = indexMap[indexMap[idx]];
            }
        }
    }
}


vector<uint> findVerticalSeam_gpu_dp(unsigned int* energies_h, unsigned int* energies_d, size_t rows, size_t cols) {
    vector<uint> seam(rows);

    // Declare pointers for device memory
    unsigned int* min_cost_d;
    unsigned int* temp_d = &(energies_d[0]);

    int row_size = cols * sizeof(int);
    int size = cols * rows * sizeof(int);

    // Allocate device memory and for inputs and outputs
    cudaMalloc((void**)&min_cost_d, size);

    // Invoke the kernel to compute the min cost table
    int cols_int = cols;
    int num_blocks = (cols - 1) / MAX_THREADS + 1;
    int num_threads = min(MAX_THREADS, cols_int);
    dim3 dim_grid(num_blocks, 1, 1);
    dim3 dim_block(num_threads, 1, 1);

    for (int row = 1; row < rows; row++)        // calculate minimum cost table row by row
        compute_min_cost_kernel << <dim_grid, dim_block >> > (energies_d, min_cost_d, temp_d, cols, rows, row); //kernel call happens height times ie no. of rows

    // Transfer result from device to host
    cudaMemcpy(energies_h, min_cost_d, size, cudaMemcpyDeviceToHost);

    // Calculate threads and blocks for a minimum reduction
    num_threads = min(nextPower2(cols), MAX_THREADS); // nextPower2 For nearest Power of 2 ie for 1029 ans. 2048
    num_blocks = (cols - 1) / num_threads + 1;
    int mins_size = num_blocks * sizeof(int);
    int min_indices_size = num_blocks * sizeof(int);
    int shared_size = num_threads * (sizeof(int) + sizeof(int));

    // Declare pointers for device and host memory
    unsigned int* row = &(energies_h[(rows - 1) * cols]);
    unsigned int* mins = (unsigned int*)malloc(mins_size);
    int* min_indices = (int*)malloc(min_indices_size);
    unsigned int* row_d;
    unsigned int* mins_d;
    int* min_indices_d;

    cudaMalloc((void**)&row_d, row_size);
    cudaMemcpy(row_d, row, row_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mins_d, mins_size);
    cudaMalloc((void**)&min_indices_d, mins_size);

    // Use the kernel function to find intermediate minimums
    find_min_kernel << <num_blocks, num_threads, shared_size >> > (row_d, mins_d, min_indices_d, cols, num_threads);

    // Compute final minimum
    cudaMemcpy(mins, mins_d, mins_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(min_indices, min_indices_d, min_indices_size, cudaMemcpyDeviceToHost);
    unsigned int minimum = mins[0];
    int min_index = min_indices[0];

    // finding minimum from bottom row
    for (int i = 1; i < num_blocks; i++) {
        if (mins[i] < minimum) {
            minimum = mins[i];
            min_index = min_indices[i];
        }
    }

    // Create the seam in reverse order. 
    seam.clear();
    seam.push_back(min_index);

    for (int i = rows - 2; i >= 1; i--) {
        unsigned int left = energies_h[(i - 1) * cols + min_index - 1];
        unsigned int middle = energies_h[(i - 1) * cols + min_index];
        unsigned int right = energies_h[(i - 1) * cols + min_index + 1];

        // Have the seam follow the least cost.
        if (left < middle && left < right) {
            min_index--; // go left
        }
        else if (right < middle && right < left) {
            min_index++; // go right
        }
        // Append to the seam.
        seam.push_back(min_index);
    }
    std::reverse(seam.begin(), seam.end());

    // Clean up.
    cudaFree(energies_d);
    cudaFree(min_cost_d);
    cudaFree(temp_d);
    cudaFree(row_d);
    cudaFree(mins_d);
    cudaFree(min_indices_d);
    return seam;
}

vector<uint> findVerticalSeam_gpu_NCSC(unsigned int* energies_h, size_t rows, size_t cols) {
    vector<uint> seam(rows);
    int ny = rows;
    int row_size = cols * sizeof(int);
    int energySize = cols * rows * sizeof(unsigned int);
    int indexMapSize = cols * (rows - 1) * sizeof(unsigned int);

    //int nStreams = 4;
    cudaStream_t streams[4];
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    // Declare pointers for device and host memory
    unsigned int* indexMap_h;
    cudaHostAlloc((void**)&indexMap_h, cols * (rows - 1) * sizeof(unsigned int), 0);

    unsigned int* energies_d;
    unsigned int* indexMap_d;
    unsigned int* offsetMap_d;
    // Allocate device memory and for inputs and outputs
    cudaMalloc((void**)&energies_d, energySize);
    cudaMalloc((void**)&indexMap_d, indexMapSize);
    cudaMalloc((void**)&offsetMap_d, indexMapSize);

    cudaMemcpyAsync(energies_d, energies_h, energySize, cudaMemcpyHostToDevice, streams[2]);

    // Invoke the kernel to build index map
    dim3 dim_block(32, 32);
    dim3 dim_grid((cols + dim_block.x - 1) / dim_block.x, (rows + dim_block.y - 1) / dim_block.y);
    build_indexMap_kernel << <dim_grid, dim_block, 0, streams[2] >> > (energies_d, indexMap_d, offsetMap_d, cols, rows);

    cudaDeviceSynchronize();
    // Transfer result from device to host
    cudaMemcpyAsync(indexMap_h, offsetMap_d, indexMapSize, cudaMemcpyDeviceToHost, streams[1]);
    //writToText2(indexMap_h, rows-1, cols, "index_cpu.txt");

    //Compute seam map
    for (int i = 2; i < rows * 2; i *= 2) {
        int num_threads_y = min(min(i, ny), MAX_THREADS);
        int num_threads_x = MAX_THREADS / num_threads_y;
        dim3 dim_block(num_threads_x, num_threads_y);
        dim3 dim_grid((cols + dim_block.x - 1) / dim_block.x, (rows + dim_block.y - 1) / dim_block.y);
        build_seamMap_kernel << <dim_grid, dim_block, 0, streams[2] >> > (energies_d, indexMap_d, cols, rows, i);
    }

    // Transfer result from device to host
    cudaMemcpyAsync(energies_h, energies_d, energySize, cudaMemcpyDeviceToHost, streams[2]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);

    // Calculate threads and blocks for a minimum reduction
    int num_threads = min(nextPower2(cols), MAX_THREADS); // nextPower2 For nearest Power of 2 ie for 1029 ans. 2048
    int num_blocks = (cols - 1) / num_threads + 1;
    int mins_size = num_blocks * sizeof(int);
    int min_indices_size = num_blocks * sizeof(int);
    int shared_size = num_threads * (sizeof(int) + sizeof(int));

    // Declare pointers for device and host memory
    //unsigned int* row = &(energies_h[0]);
    unsigned int* mins = (unsigned int*)malloc(mins_size);
    int* min_indices = (int*)malloc(min_indices_size);
    unsigned int* row_d;
    unsigned int* mins_d;
    int* min_indices_d;

    cudaMalloc((void**)&row_d, row_size);
    //cudaMemcpy(row_d, row, row_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mins_d, mins_size);
    cudaMalloc((void**)&min_indices_d, mins_size);

    // Use the kernel function to find intermediate minimums
    find_min_kernel << <num_blocks, num_threads, shared_size >> > (&(energies_d[0]), mins_d, min_indices_d, cols, num_threads);

    // Compute final minimum
    cudaMemcpy(mins, mins_d, mins_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(min_indices, min_indices_d, min_indices_size, cudaMemcpyDeviceToHost);
    unsigned int minimum = mins[0];
    int min_index = min_indices[0];

    // finding minimum from the first row
    for (int i = 1; i < num_blocks; i++) {
        if (mins[i] < minimum) {
            minimum = mins[i];
            min_index = min_indices[i];
        }
    }

    // Create the seam.
    seam.clear();
    int start = min_index;

    while (start < ((rows - 1) * cols)) {
        seam.push_back(start % cols);
        start = indexMap_h[start];
    }
    seam.push_back(start % cols);

    //Clear up
    cudaFreeHost(indexMap_h);

    cudaFree(energies_d);
    cudaFree(indexMap_d);
    cudaFree(row_d);
    cudaFree(min_indices_d);
    cudaFree(mins_d);

    return seam;
}

__global__ void removeSeam(int rows, int cols, unsigned int* seam, unsigned char* input, unsigned char* output) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows * cols) return;
    int r = index / cols, c = index % cols;
    int offset = r + (c >= seam[r]);
    index *= 3;
    offset *= 3;
    output[index] = input[index + offset];
    output[index + 1] = input[index + offset + 1];
    output[index + 2] = input[index + offset + 2];
}


__global__ void getResultImg(int rows, int cols, unsigned char* data, unsigned char* buffer) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows * cols) return;
    index = index * 3;
    data[index] = buffer[index];
    data[index + 1] = buffer[index + 1];
    data[index + 2] = buffer[index + 2];
}


vector<uint> findVerticalSeam(unsigned int* energy, size_t rows, size_t cols) {
    vector<uint> seam(rows);
    unsigned int** distTo = new unsigned int* [rows];
    short** edgeTo = new short* [rows];
    for (int i = 0; i < rows; ++i) {
        distTo[i] = new unsigned int[cols];
        edgeTo[i] = new short[cols];
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == 0)		distTo[i][j] = 0;
            else			distTo[i][j] = numeric_limits<unsigned int>::max();
            edgeTo[i][j] = 0;
        }
    }

    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col != 0)
                if (distTo[row + 1][col - 1] > distTo[row][col] + energy[(row + 1) * cols + (col - 1)]) {
                    distTo[row + 1][col - 1] = distTo[row][col] + energy[(row + 1) * cols + (col - 1)];
                    edgeTo[row + 1][col - 1] = 1;
                }
            if (distTo[row + 1][col] > distTo[row][col] + energy[(row + 1) * cols + col]) {
                distTo[row + 1][col] = distTo[row][col] + energy[(row + 1) * cols + col];
                edgeTo[row + 1][col] = 0;
            }
            if (col != cols - 1)
                if (distTo[row + 1][col + 1] > distTo[row][col] + energy[(row + 1) * cols + (col + 1)]) {
                    distTo[row + 1][col + 1] = distTo[row][col] + energy[(row + 1) * cols + (col + 1)];
                    edgeTo[row + 1][col + 1] = -1;
                }
        }
    }

    unsigned int min_index = 0, min = distTo[rows - 1][0];
    for (int i = 1; i < cols; ++i)
        if (distTo[rows - 1][i] < min) {
            min_index = i;
            min = distTo[rows - 1][i];
        }

    seam[rows - 1] = min_index;
    for (int i = rows - 1; i > 0; --i)
        seam[i - 1] = seam[i] + edgeTo[i][seam[i]];
    return seam;
}

// 1 thread <=> 1 elems, 3 pixels
__global__ void copyPixels(int rows, int cols, unsigned char* input, unsigned char* output, unsigned int* indices) {
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadIndex >= rows * cols) return;
    unsigned int index = indices[threadIndex];
    output[index * 3] = input[threadIndex * 3];
    output[index * 3 + 1] = input[threadIndex * 3 + 1];
    output[index * 3 + 2] = input[threadIndex * 3 + 2];
}

// rows, cols = input.rows, input.cols;
__global__ void initIndices(int rows, int cols, unsigned int* seam, unsigned int* indices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows * cols) return;
    indices[index] = index % cols != seam[index / cols];
}

// 1 thread <=> 2 elems
__global__ void prescan(unsigned int* data, int length, unsigned int* presum) {
    extern __shared__ unsigned int temp[]; // block.x * sizeof(unsigned int) * 2
    int base = blockIdx.x * blockDim.x * 2;
    int thid = threadIdx.x;
    int n = 2 * blockDim.x;

    int offset = 1;
    temp[2 * thid] = (2 * thid + base) < length ? data[2 * thid + base] : 0;
    temp[2 * thid + 1] = (2 * thid + 1 + base) < length ? data[2 * thid + 1 + base] : 0;

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thid == 0) temp[n - 1] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int tmp = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += tmp;
        }
    }

    __syncthreads();
    if (thid == blockDim.x - 1) {
        presum[blockIdx.x] = temp[2 * thid + 1] + data[2 * thid + 1 + base];
        // printf("presum[%d] = %d\n", blockIdx.x, presum[blockIdx.x]);
    }

    if (2 * thid + base < length) data[2 * thid + base] = temp[2 * thid];
    if (2 * thid + 1 + base < length) data[2 * thid + 1 + base] = temp[2 * thid + 1];
}

// 1 thread <=> 2 elems
__global__ void add(unsigned int* data, int length, unsigned int* presum) {
    if (blockIdx.x == 0) return;
    int base = blockIdx.x * blockDim.x * 2;
    int thid = threadIdx.x;
    if (2 * thid + base < length) data[2 * thid + base] += presum[blockIdx.x - 1];
    if (2 * thid + base + 1 < length) data[2 * thid + 1 + base] += presum[blockIdx.x - 1];
}




int ilog2(int x);
int ilog2ceil(int x);


int main() {
    Mat im = imread("D:/UofT/ECE1782/Project/seam-carving-main/pics/input/5.png");
    //imshow("image1", im);

    cout << "input size: " << im.cols << "×" << im.rows << endl;
    cout << "size: " << im.total() << endl;
    cout << "type: " << im.type() << endl; // CV_8UC3

    unsigned char* imgPtr = new unsigned char[im.rows * im.cols * im.channels()];
    unsigned char* dev_imgPtr, * dev_imgPtrBuffer;
    unsigned int* dev_energy, * dev_seams;
    int* dev_boolean, * dev_indices;

    unsigned char* cvPtr = im.ptr<unsigned char>(0);
    // size_t的取值range是目标平台下最大可能的数组尺寸
    for (size_t i = 0; i < im.rows * im.cols * im.channels(); ++i) {
        imgPtr[i] = cvPtr[i];
    }
    int paddedArraySize = 1 << ilog2ceil(im.rows * im.cols * 3);
    
    cudaMalloc((void**)&dev_imgPtr, sizeof(unsigned char) * im.rows * im.cols * im.channels());
    cudaMalloc((void**)&dev_imgPtrBuffer, sizeof(unsigned char) * im.rows * im.cols * im.channels());
    cudaMalloc((void**)&dev_energy, sizeof(unsigned int) * im.rows * im.cols);
    cudaMalloc((void**)&dev_seams, sizeof(unsigned int) * im.cols);
    cudaMalloc((void**)&dev_boolean, paddedArraySize * sizeof(int));
    cudaMalloc((void**)&dev_indices, paddedArraySize * sizeof(int));

    double timeStampA = getTimeStamp();
    cudaMemcpy(dev_imgPtr, imgPtr, sizeof(unsigned char) * im.rows * im.cols * im.channels(), cudaMemcpyHostToDevice);
    double timeStampB = getTimeStamp();
    double energyTimeCost = timeStampB - timeStampA;
    double findTimeCost = 0;
    double removeTimeCost = 0;
    


    size_t rows = im.rows;
    size_t cols = im.cols;
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1);

    for (int i = 0; i < 400; i++) {
        // calculate energy for image
        //dim3 blockEnergy(BLOCK_SIZE);
        //dim3 gridEnergy((rows * cols * 3 + blockEnergy.x - 1) / blockEnergy.x);
        double timeStampC = getTimeStamp();
        computeEnergyGPU << <gridSize, blockSize >> > (rows, cols, dev_imgPtr, dev_energy);
        //computeEnergyGPU << <gridEnergy, blockEnergy, blockEnergy.x * 3 * sizeof(char) >> > (rows, cols, dev_imgPtr, dev_energy);
        unsigned int* energy = new unsigned int[rows * cols * 3];
        double timeStampD = getTimeStamp();
        energyTimeCost += timeStampD - timeStampC;

        cudaMemcpy(energy, dev_energy, sizeof(unsigned int) * rows * cols, cudaMemcpyDeviceToHost);
        // Find vertical seam using dynamic programming
        double timeStampE = getTimeStamp();
        //vector<uint> seam = findVerticalSeam_gpu_dp(energy, dev_energy, rows, cols);
        
        //Find vertical seam using Non-Cumulative Seam Carving (NCSC) algorithm
        vector<uint> seam = findVerticalSeam_gpu_NCSC(energy,rows, cols);
        double timeStampF = getTimeStamp();
        findTimeCost += timeStampF - timeStampE;
        cudaMemcpy(dev_seams, &seam[0], sizeof(unsigned int) * rows, cudaMemcpyHostToDevice);

        // remove vertical seam

        /*
        cols--;
        dim3 block(BLOCK_SIZE);
        dim3 grid((rows * cols * 3 + block.x - 1) / block.x);
        double timeStampG = getTimeStamp();
        removeSeam << <grid, block >> > (rows, cols, dev_seams, dev_imgPtr, dev_imgPtrBuffer);
        getResultImg << <grid, block >> > (rows, cols, dev_imgPtr, dev_imgPtrBuffer);
        */

        cols--;
        dim3 block(256);
        dim3 grid((rows * cols * 3 + block.x - 1) / block.x);
        double timeStampG = getTimeStamp();
        removeSeam << <grid, block >> > (rows, cols, dev_seams, dev_imgPtr, dev_imgPtrBuffer);
        unsigned char* tmp = dev_imgPtr;
        dev_imgPtr = dev_imgPtrBuffer;
        dev_imgPtrBuffer = tmp;

        /*
        double timeStampG = getTimeStamp();
        unsigned int* dev_indices;
        cudaMalloc((void**)&dev_indices, rows * cols * sizeof(unsigned int));
        dim3 block(BLOCK_SIZE);
        dim3 grid((rows * cols + block.x - 1) / block.x);
        initIndices << <grid, block >> > (rows, cols, dev_seams, dev_indices);
        dim3 grid2((rows * cols + block.x * 2 - 1) / (block.x * 2));
        unsigned int* d_presum, * h_presum;
        cudaMalloc((void**)&d_presum, grid2.x * sizeof(unsigned int));
        cudaHostAlloc((void**)&h_presum, grid2.x * sizeof(int), 0);
        prescan << <grid2, block, block.x * sizeof(unsigned int) * 2 >> > (dev_indices, rows * cols, d_presum);
        cudaDeviceSynchronize();
        cudaMemcpy(h_presum, d_presum, grid2.x * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        for (int i = 1; i < grid2.x; i++) {
            h_presum[i] += h_presum[i - 1];
        }
        cudaMemcpy(d_presum, h_presum, grid2.x * sizeof(unsigned int), cudaMemcpyHostToDevice);
        add << <grid2, block >> > (dev_indices, rows * cols, d_presum);
        copyPixels << <grid, block >> > (rows, cols, dev_imgPtr, dev_imgPtrBuffer, dev_indices);
        cudaMemcpy(dev_imgPtr, dev_imgPtrBuffer, sizeof(unsigned char) * rows * (cols) * 3, cudaMemcpyDeviceToHost);
        cols--;
        */


        cudaDeviceSynchronize();
        double timeStampH = getTimeStamp();
        removeTimeCost += timeStampH - timeStampG;

    }


    // get the result
    double timeStampX = getTimeStamp();
    cudaMemcpy(imgPtr, dev_imgPtr, sizeof(unsigned char) * rows * (cols) * 3, cudaMemcpyDeviceToHost);
    double timeStampZ = getTimeStamp();
    double transferBackTime = timeStampZ - timeStampX;
    
    
    int sizes[2];
    sizes[0] = rows;
    sizes[1] = cols;
    // 2: 2D matrix
    Mat output(2, sizes, CV_8UC3, (void*)imgPtr);
    im = output;
    cudaFree(&dev_imgPtr);
    cudaFree(&dev_imgPtrBuffer);
    cudaFree(&dev_energy);
    cudaFree(&dev_seams);
    cudaFree(&dev_boolean);
    cudaFree(&dev_indices);
    cudaDeviceReset();


    cout << "Total energy cost time:";
    // int totaltime = energyTimeCost * 1000;
    printf("%3f s\n", energyTimeCost);
    cout << "Total find cost time:";
    printf("%3f s\n", findTimeCost);
    cout << "Total remove cost time:";
    printf("%3f s\n", removeTimeCost);
    double totalTime = energyTimeCost + findTimeCost + removeTimeCost + transferBackTime;
    cout << "Total time:"<<totalTime;
    imshow("image", im);
    imwrite("D:/UofT/ECE1782/Project/seam-carving-main/pics/outputbird1.png", im);
    waitKey(0);
    return 0;
}