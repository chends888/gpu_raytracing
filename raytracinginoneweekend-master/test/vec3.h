class vec3  {
public:
     __host__ __device__ vec3() {}
     __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2;}
     __host__ __device__ inline float x() const { return e[0]; }
     __host__ __device__ inline float y() const { return e[1]; }
     __host__ __device__ inline float z() const { return e[2]; }