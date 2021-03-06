/*****************************************************************
This is the CUDA version of backpropagation algorithm 
using GPGPU to accelerate the performance.


*****************************************************************/

#include <stdio.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------
    int layer_size, seed;
    BPNN *net_h;
    cudaError_t cuda_ret;

    // device variables
    float *input_units_d, *hidden_units_d, *output_units_d, 
        *hidden_delta_d, *output_delta_d, *target_d, *input_weights_d, 
        *hidden_weights_d, *input_prev_weights_d, *hidden_prev_weights_d;
    // __device__ int input_n_d, hidden_n_d, output_n_d;

    if(argc!=2){
        fprintf(stderr, "usage: backprop <num of input elements>\n");
        exit(0);
    }

    layer_size = atoi(argv[1]);
    seed = 7;
    bpnn_initialize(seed);
    
    printf("Setting up the problem..."); fflush(stdout);
    startTime(&timer);
    net_h = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
    load(net_h, layer_size);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("Input layer size : %d\n", layer_size);

    // Allocate device variables ----------------------------------------------

    printf("Creating device neural network..."); fflush(stdout);
    startTime(&timer);
    int in = net_h->input_n;
    int hid = net_h->hidden_n;
    int out = net_h->output_n;
    cuda_ret = cudaMalloc((void**)&(input_units_d), (in+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&hidden_units_d, (hid+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&output_units_d, (out+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&hidden_delta_d, (hid+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&output_delta_d, (out+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&target_d, (out+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&input_weights_d, (in+1)*(hid+1)*sizeof(float*));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&hidden_weights_d, (hid+1)*(out+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&input_prev_weights_d, (in+1)*(hid+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory"); 
    cuda_ret = cudaMalloc((void**)&hidden_prev_weights_d, (hid+1)*(out+1)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying neural network from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(input_units_d, net_h->input_units, (in+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(hidden_units_d, net_h->hidden_units, (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(output_units_d, net_h->output_units, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(hidden_delta_d, net_h->hidden_delta, (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(output_delta_d, net_h->output_delta, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(target_d, net_h->target, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(input_weights_d, net_h->input_weights, (in+1)*(hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(hidden_weights_d, net_h->hidden_weights, (hid+1)*(out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(input_prev_weights_d, net_h->input_prev_weights, (in+1)*(hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(hidden_prev_weights_d, net_h->hidden_prev_weights, (hid+1)*(out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Launch kernel ----------------------------------------------------------
    
    //entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    startTime(&timer);

    bpnn_train_kernel_device(in, hid, out, input_units_d, hidden_units_d, output_units_d, hidden_delta_d,
                    output_delta_d, target_d, input_weights_d, hidden_weights_d,
                    input_prev_weights_d, hidden_prev_weights_d);
    
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);
    cuda_ret = cudaMemcpy(net_h->hidden_weights, hidden_weights_d, (hid+1)*(out+1)*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the host");
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    
    // Save results to file out_gpu.txt ----------------------------------------
    const char *file_gpu = "out_gpu.txt";
    bpnn_save_dbg(net_h, file_gpu);

    // Verify correctness -----------------------------------------------------
    // const chat *file_cpu = "../out.txt";
    // compare_result(file_gpu, file_cpu);

    // Free memory ------------------------------------------------------------
    bpnn_free(net_h);
    cudaFree(input_units_d); cudaFree(hidden_units_d); cudaFree(output_units_d);
    cudaFree(hidden_delta_d); cudaFree(output_delta_d); cudaFree(target_d);
    cudaFree(input_weights_d); cudaFree(hidden_weights_d); 
    cudaFree(input_prev_weights_d); cudaFree(hidden_prev_weights_d); 
    printf("Training done!\n\n");

    return 0;
}

