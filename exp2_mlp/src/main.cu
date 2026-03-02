#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp_layers.cuh"

struct Options {
    std::vector<int> layers = {1024, 2048, 1024};  // includes input dim and final output dim
    int batch = 128;
    std::string activation = "relu";
    std::string impl = "baseline";  // baseline | activation_fused
    bool verify = true;
};

std::vector<int> parse_layers_list(const std::string& csv) {
    std::vector<int> dims;
    size_t start = 0;
    while (start < csv.size()) {
        size_t comma = csv.find(',', start);
        const size_t len = (comma == std::string::npos) ? (csv.size() - start) : (comma - start);
        if (len > 0) {
            dims.push_back(std::stoi(csv.substr(start, len)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return dims;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = parse_layers_list(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            opt.activation = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dmlp --layers 1024,2048,1024 --batch 128 --activation relu \\\n  --impl baseline|activation_fused [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.layers.size() < 2) {
        throw std::invalid_argument("--layers must contain at least two integers (input/output)");
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void seed_tensor(std::vector<float>& data, float scale) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = scale * std::sin(0.11f * static_cast<float>(i));
    }
}

void mlp_cpu_reference(const std::vector<int>& layers,
                       int batch,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       const std::vector<size_t>& weight_offsets,
                       const std::vector<size_t>& bias_offsets,
                       const std::vector<float>& input,
                       std::vector<float>& output,
                       const std::string& activation) {
    const int num_layers = static_cast<int>(layers.size()) - 1;
    std::vector<float> current(input);
    std::vector<float> next;
    
    for (int layer = 0; layer < num_layers; ++layer) {
        const int in_dim = layers[layer];
        const int out_dim = layers[layer + 1];
        next.resize(batch * out_dim, 0.0f);
        
        // GEMM: output = input @ weight^T (weight is [out_dim, in_dim])
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_dim; ++o) {
                float sum = 0.0f;
                for (int i = 0; i < in_dim; ++i) {
                    sum += current[b * in_dim + i] * weights[weight_offsets[layer] + o * in_dim + i];
                }
                next[b * out_dim + o] = sum;
            }
        }
        
        // Add bias and apply activation
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_dim; ++o) {
                int idx = b * out_dim + o;
                float val = next[idx] + biases[bias_offsets[layer] + o];
                
                // Apply activation
                if (activation == "relu") {
                    val = std::max(0.0f, val);
                } else if (activation == "gelu") {
                    const float sqrt_2_over_pi = 0.7978845608f;
                    float x_cubed = val * val * val;
                    float inner = sqrt_2_over_pi * (val + 0.044715f * x_cubed);
                    val = 0.5f * val * (1.0f + std::tanh(inner));
                }
                
                next[idx] = val;
            }
        }
        
        current = next;
    }
    
    output = current;
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int batch = opt.batch;
    const size_t input_elems = static_cast<size_t>(batch) * opt.layers.front();
    const size_t output_elems = static_cast<size_t>(batch) * opt.layers.back();
    const int num_layers = static_cast<int>(opt.layers.size()) - 1;

    std::vector<size_t> weight_offsets(num_layers, 0);
    std::vector<size_t> bias_offsets(num_layers, 0);
    size_t weight_cursor = 0;
    size_t bias_cursor = 0;
    for (int i = 0; i < num_layers; ++i) {
        const int in_dim = opt.layers[i];
        const int out_dim = opt.layers[i + 1];
        weight_offsets[i] = weight_cursor;
        bias_offsets[i] = bias_cursor;
        weight_cursor += static_cast<size_t>(out_dim) * in_dim;
        bias_cursor += static_cast<size_t>(out_dim);
    }

    std::vector<float> h_input(input_elems);
    std::vector<float> h_weights(weight_cursor);
    std::vector<float> h_biases(bias_cursor);
    std::vector<float> h_output(output_elems, 0.0f);
    std::vector<float> h_ref(output_elems, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weights, 0.25f);
    seed_tensor(h_biases, 0.01f);

    float* d_input = nullptr;
    float* d_workspace_a = nullptr;
    float* d_workspace_b = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;
    
    // Determine maximum activation buffer size needed
    size_t max_activation_elems = input_elems;
    for (int i = 0; i < num_layers; ++i) {
        size_t layer_elems = static_cast<size_t>(batch) * opt.layers[i + 1];
        max_activation_elems = std::max(max_activation_elems, layer_elems);
    }
    
    // Allocate device buffers
    check_cuda(cudaMalloc(&d_input, input_elems * sizeof(float)), "malloc d_input");
    check_cuda(cudaMalloc(&d_workspace_a, max_activation_elems * sizeof(float)), "malloc d_workspace_a");
    check_cuda(cudaMalloc(&d_workspace_b, max_activation_elems * sizeof(float)), "malloc d_workspace_b");
    check_cuda(cudaMalloc(&d_weights, weight_cursor * sizeof(float)), "malloc d_weights");
    check_cuda(cudaMalloc(&d_biases, bias_cursor * sizeof(float)), "malloc d_biases");
    
    // Copy data to device
    check_cuda(cudaMemcpy(d_input, h_input.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemcpy(d_weights, h_weights.data(), weight_cursor * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
    check_cuda(cudaMemcpy(d_biases, h_biases.data(), bias_cursor * sizeof(float), cudaMemcpyHostToDevice), "copy biases");
    
    // Initialize first workspace with input
    check_cuda(cudaMemcpy(d_workspace_a, d_input, input_elems * sizeof(float), cudaMemcpyDeviceToDevice), "copy input to workspace");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start, stream), "record baseline start");
        for (int layer = 0; layer < num_layers; ++layer) {
            LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
            const float* d_w = d_weights + weight_offsets[layer];
            const float* d_b = d_biases + bias_offsets[layer];
            run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle);
            launch_bias_add(d_b, d_workspace_b, shape, stream);
            launch_activation(opt.activation, d_workspace_b, shape, stream);
            std::swap(d_workspace_a, d_workspace_b);
        }
        check_cuda(cudaEventRecord(stop, stream), "record baseline stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline");
    } else if (opt.impl == "activation_fused") {
        check_cuda(cudaEventRecord(start, stream), "record fused start");
        for (int layer = 0; layer < num_layers; ++layer) {
            LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
            const float* d_w = d_weights + weight_offsets[layer];
            const float* d_b = d_biases + bias_offsets[layer];
            run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle);
            launch_fused_bias_activation(d_b, opt.activation, d_workspace_b, shape, stream);
            std::swap(d_workspace_a, d_workspace_b);
        }
        check_cuda(cudaEventRecord(stop, stream), "record fused stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
    } else {
        throw std::invalid_argument("Unknown --impl " + opt.impl);
    }

    // Copy final activations back to host
    // After the loop, result is in d_workspace_a due to the final swap
    check_cuda(cudaMemcpy(h_output.data(), d_workspace_a, output_elems * sizeof(float), cudaMemcpyDeviceToHost), "copy output");

    if (opt.verify) {
        mlp_cpu_reference(opt.layers,
                          batch,
                          h_weights,
                          h_biases,
                          weight_offsets,
                          bias_offsets,
                          h_input,
                          h_ref,
                          opt.activation);
        
        // Compute max absolute difference
        float max_error = 0.0f;
        for (size_t i = 0; i < h_output.size(); ++i) {
            float err = std::abs(h_output[i] - h_ref[i]);
            max_error = std::max(max_error, err);
        }
        std::cout << "Max error vs CPU: " << std::scientific << max_error << std::endl;
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " Batch=" << batch << " Layers=";
        for (size_t i = 0; i < opt.layers.size(); ++i) {
            std::cout << opt.layers[i];
            if (i + 1 < opt.layers.size()) {
                std::cout << "x";
            }
        }
        std::cout << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << mlp_gflops(opt.layers, batch, elapsed_ms) << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    // Cleanup
    check_cuda(cudaFree(d_input), "free d_input");
    check_cuda(cudaFree(d_workspace_a), "free d_workspace_a");
    check_cuda(cudaFree(d_workspace_b), "free d_workspace_b");
    check_cuda(cudaFree(d_weights), "free d_weights");
    check_cuda(cudaFree(d_biases), "free d_biases");
    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    check_cublas(cublasDestroy(handle), "destroy handle");
    
    return 0;
}
