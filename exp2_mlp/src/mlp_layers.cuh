#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

struct LayerShape {
    int batch;
    int in_dim;
    int out_dim;
};

inline double layer_flops(const LayerShape& shape) {
    return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int>& layers, int batch, double millis) {
    double total_flops = 0.0;
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        LayerShape shape{batch, layers[i], layers[i + 1]};
        total_flops += layer_flops(shape);
    }
    return total_flops / (millis * 1e6);
}

__global__ void bias_add_kernel(const float* __restrict__ bias,
                                float* __restrict__ activations,
                                LayerShape shape) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    
    if (idx < total_elements) {
        // Determine which neuron this element corresponds to
        int neuron_idx = idx % shape.out_dim;
        activations[idx] += bias[neuron_idx];
    }
}

__global__ void relu_kernel(float* __restrict__ activations, size_t elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        activations[idx] = fmaxf(0.0f, activations[idx]);
    }
}

__global__ void gelu_kernel(float* __restrict__ activations, size_t elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        float x = activations[idx];
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        activations[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

inline void launch_bias_add(const float* bias, float* activations, const LayerShape& shape, cudaStream_t stream) {
    const int threads = 256;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
    (void)elements;  // silence unused warnings until kernel implemented
}

inline void launch_activation(const std::string& activation,
                              float* activations,
                              const LayerShape& shape,
                              cudaStream_t stream) {
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (activation == "relu") {
        relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else if (activation == "gelu") {
        gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else {
        // TODO(student): add more activations as desired
    }
}

__global__ void fused_bias_activation_kernel(const float* __restrict__ bias,
                                             float* __restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    
    if (idx < total_elements) {
        // Add bias
        int neuron_idx = idx % shape.out_dim;
        float val = activations[idx] + bias[neuron_idx];
        
        // Apply activation
        if (activation_type == 0) {  // ReLU
            val = fmaxf(0.0f, val);
        } else if (activation_type == 1) {  // GELU
            const float sqrt_2_over_pi = 0.7978845608f;
            float x_cubed = val * val * val;
            float inner = sqrt_2_over_pi * (val + 0.044715f * x_cubed);
            val = 0.5f * val * (1.0f + tanhf(inner));
        }
        
        activations[idx] = val;
    }
}

inline void launch_fused_bias_activation(const float* bias,
                                         const std::string& activation,
                                         float* activations,
                                         const LayerShape& shape,
                                         cudaStream_t stream) {
    int activation_type = (activation == "gelu") ? 1 : 0;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape, activation_type);
    (void)elements;
}

inline void run_gemm_layer(const float* input,
                           const float* weight,
                           float* output,
                           const LayerShape& shape,
                           cublasHandle_t handle) {
    // Compute: output[batch, out_dim] = input[batch, in_dim] @ weight^T[in_dim, out_dim]
    // Weight is stored row-major as [out_dim, in_dim]
    // We need: C = A * B^T where A is input, B is weight
    // In cuBLAS column-major: C = B * A^T (with transpose flags swapped)
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cublasSgemm computes: C = alpha * op(A) * op(B) + beta * C
    // With row-major to column-major conversion:
    // output[batch, out_dim] = input[batch, in_dim] @ weight[out_dim, in_dim]^T
    cublasSgemm(handle,
                CUBLAS_OP_T,      // transpose weight: [out_dim, in_dim] -> [in_dim, out_dim]
                CUBLAS_OP_N,      // no transpose on input
                shape.out_dim,    // rows of output (columns in col-major)
                shape.batch,      // cols of output (rows in col-major)
                shape.in_dim,     // inner dimension
                &alpha,
                weight,           // [out_dim, in_dim] stored row-major
                shape.in_dim,     // leading dimension of weight
                input,            // [batch, in_dim] stored row-major
                shape.in_dim,     // leading dimension of input
                &beta,
                output,           // [batch, out_dim] stored row-major
                shape.out_dim);   // leading dimension of output
}
