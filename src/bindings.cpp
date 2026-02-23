#include <torch/extension.h>

// Existing physics bridge
void compute_mesh_associator_force_cuda(torch::Tensor q, torch::Tensor f, float stiffness, int N_x, int N_y, int N_z);

// New crypto bridge
void batched_sedenion_mul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void compute_mesh_associator_force(torch::Tensor q, torch::Tensor f, float stiffness) {
    // ... (keep your existing checks here) ...
    int N_x = q.size(0); int N_y = q.size(1); int N_z = q.size(2);
    compute_mesh_associator_force_cuda(q, f, stiffness, N_x, N_y, N_z);
}

void batched_sedenion_mul(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.device().is_cuda(), "c must be a CUDA tensor");
    TORCH_CHECK(a.size(1) == 16 && b.size(1) == 16 && c.size(1) == 16, "Tensors must be 16D for Sedenions");
    
    batched_sedenion_mul_cuda(a, b, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mesh_associator_force", &compute_mesh_associator_force, "APH Rank-5 Spatial Physics");
    m.def("batched_sedenion_mul", &batched_sedenion_mul, "Synergeia Rank-5 Cryptographic Multiplier");
}