// Código adaptado de:
// https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/
#include <iostream>
#include <fstream>
#include <float.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}


// Defining GPU kernel
__global__ void render(vec3 *fb, int width, int height, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // Check if block id is within block size
    if ((i < width) && (j < height)) {
        int pixel_index = j*width + i;
        float u = float(i) / float(width);
        float v = float(j) / float(height);
        ray r(origin, lower_left_corner + u*horizontal + v*vertical);
        fb[pixel_index] = color(r, world);
    }

}

__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Generate two spheres
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-1,-1), 0.5);
        *d_world    = new hitable_list(d_list,2);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}



int main() {
    int nx = 100;
    int ny = 50;
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(vec3);
    vec3 *fb;
    // Allocate frame buffer for individual pixels in GPU
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Create world of hitables
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Define number of threads per block
    int tx = 8;
    int ty = 8;
    dim3 threads(tx,ty);
    // Define block size
    dim3 blocks(nx/tx+1,ny/ty+1);
    // Render image on GPU
    render<<<blocks, threads>>>(fb, nx, ny,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0),
                                d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::ofstream myfile;
    myfile.open ("image.ppm");
    myfile << "P3\n" << nx << " " << ny << "\n255\n";
    // Get pixels value
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(25*fb[pixel_index].r());
            int ig = int(25*fb[pixel_index].g());
            int ib = int(25*fb[pixel_index].b());
            // std::cout << ir << " " << ig << " " << ib << "\n";
            myfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    // Free used memory and check errors
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
