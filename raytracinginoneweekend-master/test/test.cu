#include <iostream>
#include <fstream>
#include "vec3.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
__global__ void render(vec3 *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    fb[pixel_index] = vec3( float(i) / max_x, float(j) / max_y, 0.2f);
}


int main() {
    int nx = 100;
    int ny = 50;
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    // allocate FB
    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));



    int tx = 8;
    int ty = 8;
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    std::ofstream myfile;
    myfile.open ("image.ppm");
    myfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            float r = float(i) / float (nx);
            float g = float(j) / float (ny);
            float b = 0.2;
            int ir = int(255.99*r); 
            int ig = int(255.99*g); 
            int ib = int(255.99*b); 
            // std::cout << ir << " " << ig << " " << ib << "\n";
            myfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
}