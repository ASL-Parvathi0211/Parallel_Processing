#include <cuda_runtime.h> // Include the CUDA runtime library
#include <algorithm>      // Include the algorithm library for standard algorithms
#include <iostream>       // Include the iostream library for input and output

// Kernel function to update the matrix
__global__ void updateMatrixKernel(float *d_D, float *d_vbuf, float *d_hbuf, int n, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // Calculate the thread ID
    int numThreads = gridDim.x * blockDim.x;         // Calculate the total number of threads
    
    // Each thread updates part of vbuf
    for (int i = tid; i < n; i += numThreads) {
        d_vbuf[i] = d_D[i * n + k];
    }
    
    __syncthreads(); // Synchronize threads to ensure all vbuf updates are done

    // Each thread updates part of hbuf
    for (int j = tid; j < n; j += numThreads) {
        d_hbuf[j] = d_D[k * n + j];
    }

    __syncthreads(); // Synchronize threads to ensure all hbuf updates are done

    // Each thread updates part of the matrix
    for (int i = tid; i < n; i += numThreads) {
        for (int j = 0; j < n; ++j) {
            d_D[i * n + j] = fminf(d_D[i * n + j], d_vbuf[i] + d_hbuf[j]);
        }
    }
}

// Function to update the matrix
void updateMatrix(float *D, int n) {
    float *d_D, *d_vbuf, *d_hbuf; // Pointers for device memory

    size_t matrixSize = n * n * sizeof(float); // Calculate size of the matrix
    cudaMalloc((void**)&d_D, matrixSize);      // Allocate device memory for the matrix
    cudaMemcpy(d_D, D, matrixSize, cudaMemcpyHostToDevice); // Copy matrix to device memory

    cudaMalloc((void**)&d_vbuf, n * sizeof(float)); // Allocate device memory for vbuf
    cudaMalloc((void**)&d_hbuf, n * sizeof(float)); // Allocate device memory for hbuf

    int numThreads = 4;    // Number of threads per block
    int numBlocks = (n + numThreads - 1) / numThreads; // Calculate number of blocks

    // Loop over each k value
    for (int k = 0; k < n; ++k) {
        updateMatrixKernel<<<numBlocks, numThreads>>>(d_D, d_vbuf, d_hbuf, n, k); // Launch kernel
        cudaDeviceSynchronize(); // Synchronize device
    }

    cudaMemcpy(D, d_D, matrixSize, cudaMemcpyDeviceToHost); // Copy result back to host memory

    cudaFree(d_D);   // Free device memory for matrix
    cudaFree(d_vbuf);// Free device memory for vbuf
    cudaFree(d_hbuf);// Free device memory for hbuf
}

// Function to print the matrix
void printMatrix(float *D, int n, const char* label) {
    std::cout << label << std::endl; // Print label
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << D[i * n + j] << " "; // Print each element of the matrix
        }
        std::cout << std::endl; // Newline after each row
    }
}

int main() {
    int n = 4; // Size of the matrix
    float *D = new float[n * n]; // Allocate memory for the matrix on the host

    // Initialize matrix D with example values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                D[i * n + j] = 0.0f;  // Set diagonal to 0
            } else {
                D[i * n + j] = static_cast<float>(rand() % 100 + 1);  // Positive values elsewhere
            }
        }
    }

    printMatrix(D, n, "Initial Matrix:"); // Print initial matrix

    updateMatrix(D, n); // Update matrix

    printMatrix(D, n, "Final Matrix:"); // Print final matrix

    delete[] D; // Free host memory
    return 0;   // Return success
}
