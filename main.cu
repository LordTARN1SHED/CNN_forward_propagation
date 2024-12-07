#include "kernel.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

int main()
{
    // 假设的尺寸
    int inputSize = 1024;                        // 输入大小32*32
    int filterSize = 3;                          // 过滤器大小
    int outputSize = inputSize - filterSize + 1; // 输出大小

    // 分配主机内存
    float *h_input = (float *)malloc(inputSize * inputSize * sizeof(float));
    float *h_filter = (float *)malloc(filterSize * filterSize * sizeof(float));
    float *h_output = (float *)malloc(outputSize * outputSize * sizeof(float));

    // 初始化输入和过滤器（这里简化为随机值）
    for (int i = 0; i < inputSize * inputSize; i++)
    {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < filterSize * filterSize; i++)
    {
        h_filter[i] = rand() / (float)RAND_MAX;
    }

    // 分配设备内存
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_filter, filterSize * filterSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    // 复制数据到设备
    cudaMemcpy(d_input, h_input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);

    // 调用卷积核函数
    dim3 blockDim(16, 16);
    dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x, (outputSize + blockDim.y - 1) / blockDim.y);
    convolve<<<gridDim, blockDim>>>(d_input, d_output, d_filter, inputSize, filterSize);

    // 调用ReLU核函数
    int reluBlockSize = 256;
    int reluGridSize = (outputSize * outputSize + reluBlockSize - 1) / reluBlockSize;
    relu<<<reluGridSize, reluBlockSize>>>(d_output, outputSize * outputSize);

    // 假设目标输出和损失函数计算
    float *d_target, *d_loss;
    cudaMalloc(&d_target, outputSize * outputSize * sizeof(float));
    cudaMalloc(&d_loss, outputSize * outputSize * sizeof(float));
    // 初始化d_target（省略）
    mse_loss<<<reluGridSize, reluBlockSize>>>(d_output, d_target, d_loss, outputSize * outputSize);

    // 反向传播和权重更新（省略具体实现）

    // 反向传播
    // 假设损失函数梯度已计算
    // d_loss_grad, d_relu_grad, d_conv_grad的定义和分配
    // 假设的梯度数组和相关尺寸
    float *d_loss_grad, *d_relu_grad, *d_conv_grad;
    cudaMalloc(&d_loss_grad, outputSize * outputSize * sizeof(float));
    cudaMalloc(&d_relu_grad, outputSize * outputSize * sizeof(float));
    cudaMalloc(&d_conv_grad, filterSize * filterSize * sizeof(float));

    // 通过ReLU层反向传播梯度
    relu_backward<<<reluGridSize, reluBlockSize>>>(d_output, d_loss_grad, d_relu_grad, outputSize * outputSize);

    // 计算卷积层的权重梯度
    conv_backward<<<gridDim, blockDim>>>(d_input, d_relu_grad, d_conv_grad, inputSize, filterSize);

    // 权重更新
    float learning_rate = 0.01; // 学习率
    update_weights<<<1, filterSize * filterSize>>>(d_filter, d_conv_grad, filterSize * filterSize, learning_rate);

    // 复制结果回主机
    cudaMemcpy(h_output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果（省略）

    // 清理
    // 最后，释放主机和设备上的内存
    free(hostInputData);
    cudaFree(deviceInput);
    cudaFree(fcWeights);
    // ... 释放其他分配的内存 ...
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_loss);
    free(h_input);
    free(h_filter);
    free(h_output);


    return 0;
}
