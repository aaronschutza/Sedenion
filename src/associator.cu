#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Oct8 { float e0, e1, e2, e3, e4, e5, e6, e7; };
struct Sed16 { Oct8 a; Oct8 b; };

// --- OCTONION BASE LOGIC ---
__device__ Oct8 octonion_mul(Oct8 a, Oct8 b) {
    Oct8 c;
    c.e0 = a.e0*b.e0 - a.e1*b.e1 - a.e2*b.e2 - a.e3*b.e3 - a.e4*b.e4 - a.e5*b.e5 - a.e6*b.e6 - a.e7*b.e7;
    c.e1 = a.e0*b.e1 + a.e1*b.e0 + a.e2*b.e4 + a.e3*b.e7 - a.e4*b.e2 + a.e5*b.e6 - a.e6*b.e5 - a.e7*b.e3;
    c.e2 = a.e0*b.e2 - a.e1*b.e4 + a.e2*b.e0 + a.e3*b.e5 + a.e4*b.e1 - a.e5*b.e3 + a.e6*b.e7 - a.e7*b.e6;
    c.e3 = a.e0*b.e3 - a.e1*b.e7 - a.e2*b.e5 + a.e3*b.e0 + a.e4*b.e6 + a.e5*b.e2 - a.e6*b.e4 + a.e7*b.e1;
    c.e4 = a.e0*b.e4 + a.e1*b.e2 - a.e2*b.e1 - a.e3*b.e6 + a.e4*b.e0 + a.e5*b.e7 + a.e6*b.e3 - a.e7*b.e5;
    c.e5 = a.e0*b.e5 - a.e1*b.e6 + a.e2*b.e3 - a.e3*b.e2 - a.e4*b.e7 + a.e5*b.e0 + a.e6*b.e1 + a.e7*b.e4;
    c.e6 = a.e0*b.e6 + a.e1*b.e5 - a.e2*b.e7 + a.e3*b.e4 - a.e4*b.e3 - a.e5*b.e1 + a.e6*b.e0 + a.e7*b.e2;
    c.e7 = a.e0*b.e7 + a.e1*b.e3 + a.e2*b.e6 - a.e3*b.e1 + a.e4*b.e5 - a.e5*b.e4 - a.e6*b.e2 + a.e7*b.e0;
    return c;
}

__device__ Oct8 octonion_conj(Oct8 a) {
    return {a.e0, -a.e1, -a.e2, -a.e3, -a.e4, -a.e5, -a.e6, -a.e7};
}

__device__ Oct8 octonion_add(Oct8 a, Oct8 b) {
    return {a.e0+b.e0, a.e1+b.e1, a.e2+b.e2, a.e3+b.e3, a.e4+b.e4, a.e5+b.e5, a.e6+b.e6, a.e7+b.e7};
}

__device__ Oct8 octonion_sub(Oct8 a, Oct8 b) {
    return {a.e0-b.e0, a.e1-b.e1, a.e2-b.e2, a.e3-b.e3, a.e4-b.e4, a.e5-b.e5, a.e6-b.e6, a.e7-b.e7};
}

// --- SEDENION LOGIC (CAYLEY-DICKSON) ---
__device__ Sed16 sedenion_mul(Sed16 x, Sed16 y) {
    Sed16 z;
    // z.a = x.a * y.a - y.b^* * x.b
    z.a = octonion_sub(octonion_mul(x.a, y.a), octonion_mul(octonion_conj(y.b), x.b));
    // z.b = y.b * x.a + x.b * y.a^*
    z.b = octonion_add(octonion_mul(y.b, x.a), octonion_mul(x.b, octonion_conj(y.a)));
    return z;
}

__device__ Sed16 sedenion_associator(Sed16 a, Sed16 b, Sed16 c) {
    Sed16 ab = sedenion_mul(a, b);
    Sed16 bc = sedenion_mul(b, c);
    Sed16 ab_c = sedenion_mul(ab, c);
    Sed16 a_bc = sedenion_mul(a, bc);
    
    Sed16 assoc;
    assoc.a = octonion_sub(ab_c.a, a_bc.a);
    assoc.b = octonion_sub(ab_c.b, a_bc.b);
    return assoc;
}

// --- 3D 16-DIMENSIONAL KERNEL ---
__global__ void compute_volumetric_sedenion_kernel(
    const float* __restrict__ q, 
    float* __restrict__ f, 
    float stiffness,
    int N_x, int N_y, int N_z) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_nodes = N_x * N_y * N_z;
    if (idx >= total_nodes) return;

    int x = idx % N_x;
    int y = (idx / N_x) % N_y;
    int z = idx / (N_x * N_y);

    int px = (x > 0 ? x - 1 : 0) + y * N_x + z * N_x * N_y;
    int nx = (x < N_x - 1 ? x + 1 : N_x - 1) + y * N_x + z * N_x * N_y;
    int py = x + (y > 0 ? y - 1 : 0) * N_x + z * N_x * N_y;
    int ny = x + (y < N_y - 1 ? y + 1 : N_y - 1) * N_x + z * N_x * N_y;
    int pz = x + y * N_x + (z > 0 ? z - 1 : 0) * N_x * N_y;
    int nz = x + y * N_x + (z < N_z - 1 ? z + 1 : N_z - 1) * N_x * N_y;

    auto load_sed16 = [&](int i) -> Sed16 {
        return {
            {q[i*16+0], q[i*16+1], q[i*16+2], q[i*16+3], q[i*16+4], q[i*16+5], q[i*16+6], q[i*16+7]},
            {q[i*16+8], q[i*16+9], q[i*16+10], q[i*16+11], q[i*16+12], q[i*16+13], q[i*16+14], q[i*16+15]}
        };
    };

    Sed16 q_i = load_sed16(idx);
    
    Sed16 a_x = sedenion_associator(load_sed16(px), q_i, load_sed16(nx));
    Sed16 a_y = sedenion_associator(load_sed16(py), q_i, load_sed16(ny));
    Sed16 a_z = sedenion_associator(load_sed16(pz), q_i, load_sed16(nz));

    // Force extraction across all 16 dimensions
    f[idx*16+0]  = -stiffness * (a_x.a.e0 + a_y.a.e0 + a_z.a.e0);
    f[idx*16+1]  = -stiffness * (a_x.a.e1 + a_y.a.e1 + a_z.a.e1);
    f[idx*16+2]  = -stiffness * (a_x.a.e2 + a_y.a.e2 + a_z.a.e2);
    f[idx*16+3]  = -stiffness * (a_x.a.e3 + a_y.a.e3 + a_z.a.e3);
    f[idx*16+4]  = -stiffness * (a_x.a.e4 + a_y.a.e4 + a_z.a.e4);
    f[idx*16+5]  = -stiffness * (a_x.a.e5 + a_y.a.e5 + a_z.a.e5);
    f[idx*16+6]  = -stiffness * (a_x.a.e6 + a_y.a.e6 + a_z.a.e6);
    f[idx*16+7]  = -stiffness * (a_x.a.e7 + a_y.a.e7 + a_z.a.e7);
    
    f[idx*16+8]  = -stiffness * (a_x.b.e0 + a_y.b.e0 + a_z.b.e0);
    f[idx*16+9]  = -stiffness * (a_x.b.e1 + a_y.b.e1 + a_z.b.e1);
    f[idx*16+10] = -stiffness * (a_x.b.e2 + a_y.b.e2 + a_z.b.e2);
    f[idx*16+11] = -stiffness * (a_x.b.e3 + a_y.b.e3 + a_z.b.e3);
    f[idx*16+12] = -stiffness * (a_x.b.e4 + a_y.b.e4 + a_z.b.e4);
    f[idx*16+13] = -stiffness * (a_x.b.e5 + a_y.b.e5 + a_z.b.e5);
    f[idx*16+14] = -stiffness * (a_x.b.e6 + a_y.b.e6 + a_z.b.e6);
    f[idx*16+15] = -stiffness * (a_x.b.e7 + a_y.b.e7 + a_z.b.e7);
}

void compute_mesh_associator_force_cuda(torch::Tensor q, torch::Tensor f, float stiffness, int N_x, int N_y, int N_z) {
    int total_nodes = N_x * N_y * N_z;
    int threads = 256;
    int blocks = (total_nodes + threads - 1) / threads;
    compute_volumetric_sedenion_kernel<<<blocks, threads>>>(q.data_ptr<float>(), f.data_ptr<float>(), stiffness, N_x, N_y, N_z);
}
// ========================================================================
// SYNERGEIA: BATCHED CRYPTOGRAPHIC KERNEL (1D TENSOR GRAPH)
// ========================================================================

__global__ void batched_sedenion_mul_kernel(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c, 
    int num_transactions) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_transactions) return;

    // Load 16D states for Transaction A and Lock B
    Sed16 x = {
        {a[idx*16+0], a[idx*16+1], a[idx*16+2], a[idx*16+3], a[idx*16+4], a[idx*16+5], a[idx*16+6], a[idx*16+7]},
        {a[idx*16+8], a[idx*16+9], a[idx*16+10], a[idx*16+11], a[idx*16+12], a[idx*16+13], a[idx*16+14], a[idx*16+15]}
    };
    
    Sed16 y = {
        {b[idx*16+0], b[idx*16+1], b[idx*16+2], b[idx*16+3], b[idx*16+4], b[idx*16+5], b[idx*16+6], b[idx*16+7]},
        {b[idx*16+8], b[idx*16+9], b[idx*16+10], b[idx*16+11], b[idx*16+12], b[idx*16+13], b[idx*16+14], b[idx*16+15]}
    };

    // Execute Native Sedenion Trapdoor
    Sed16 z = sedenion_mul(x, y);

    // Write back to the output tensor
    c[idx*16+0] = z.a.e0; c[idx*16+1] = z.a.e1; c[idx*16+2] = z.a.e2; c[idx*16+3] = z.a.e3;
    c[idx*16+4] = z.a.e4; c[idx*16+5] = z.a.e5; c[idx*16+6] = z.a.e6; c[idx*16+7] = z.a.e7;
    c[idx*16+8] = z.b.e0; c[idx*16+9] = z.b.e1; c[idx*16+10]= z.b.e2; c[idx*16+11]= z.b.e3;
    c[idx*16+12]= z.b.e4; c[idx*16+13]= z.b.e5; c[idx*16+14]= z.b.e6; c[idx*16+15]= z.b.e7;
}

// The PyBind Bridge
void batched_sedenion_mul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int num_transactions = a.size(0);
    int threads = 256;
    int blocks = (num_transactions + threads - 1) / threads;
    batched_sedenion_mul_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), num_transactions);
}