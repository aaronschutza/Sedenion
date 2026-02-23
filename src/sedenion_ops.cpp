#include <torch/extension.h>

// Forward declaration of the CUDA launcher
void compute_mesh_associator_force_cuda(
    torch::Tensor q, 
    torch::Tensor f, 
    float stiffness, 
    int N_x, 
    int N_y);

// C++ Interface Validation
void compute_mesh_associator_force(
    torch::Tensor q, 
    torch::Tensor f, 
    float stiffness) 
{
    // 1. Hardware and Memory Checks
    TORCH_CHECK(q.device().is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(f.device().is_cuda(), "f must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous in memory");
    TORCH_CHECK(f.is_contiguous(), "f must be contiguous in memory");
    
    // 2. Dimensionality Checks (The Cortical Sheet)
    TORCH_CHECK(q.dim() == 3, "q must be a 3D tensor [N_x, N_y, 7]");
    TORCH_CHECK(q.size(2) == 7, "The final dimension must be 7 (The Imaginary Octonions)");

    // 3. Extract Macro-Architecture Dimensions
    int N_x = q.size(0);
    int N_y = q.size(1);

    // Fire the kernel
    compute_mesh_associator_force_cuda(q, f, stiffness, N_x, N_y);
}

// PyBind11 Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mesh_associator_force", &compute_mesh_associator_force, "APH 2D Cortical Mesh Associator Force (CUDA)");
}