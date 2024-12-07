
// ###########################//

// ######### 前向传播 #########//

// ###########################//

#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <curand.h>
// #include <curand_kernel.h>

// 假设的一些网络参数
const int inputSize = 224;                               // 假设输入是 224x224
const int filterSize = 3;                                // 假设卷积核大小为 3x3
const int numFilters = 16;                               // 假设有 16 个过滤器
const int poolSize = 2;                                  // 假设池化大小为 2x2
const int convOutputSize = (inputSize - filterSize + 1); // 卷积输出大小
const int pooledOutputSize = convOutputSize / poolSize;  // 池化输出大小
const int fcOutputSize = 128;                            // 假设全连接层输出大小为 128
const int finalOutputSize = 10;                          // 假设最终输出大小为 10 (例如，10类分类问题)
const int batchSize = 128;                               // 示例批次大小
const int stride = 1, padding = 0;                       // 卷积步长为1，边界扩展为0
const int ConvTime = 5, FcTime = 3;                      // 五层卷积，三层全链接
const int typenum = 10;                                  // 最后分类的种类个数

// 卷积核函数
__global__ void convolve(float *input, float *output, float *filter, int inputWidth, int inputHeight, int filterWidth, int filterHeight, int stride, int padding, int batchSize)
{
    // 额这里的传参有点多，我稍微解释一下：
    // input：传入矩阵的指针  output传出矩阵的指针  filter：之前初始化完毕的过滤器指针  inputWidth inputHeight：实际上两个值一样因为是正方形但是每层会变  filterWidth filterHeight：都是3  stride：步长  padding：边界填充的大小  batchSize：批大小

    extern __shared__ float tile[]; // 利用extern关键字可以动态的分配数组大小，tile实际上是二位的数据但是便于后续的全连接层本代码中所有的二维数组均转换为一维数组使用

    int batchIndex = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int outputHeight = (inputHeight - filterHeight + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - filterWidth + 2 * padding) / stride + 1;

    // 计算tile中的相对位置
    int rowInTile = ty * stride - padding;
    int colInTile = tx * stride - padding;

    // 加载tile到共享内存
    for (int i = 0; i < filterHeight; i += stride)
    {
        for (int j = 0; j < filterWidth; j += stride)
        {
            int rowToLoad = rowInTile + i;
            int colToLoad = colInTile + j;
            if (rowToLoad >= 0 && rowToLoad < inputHeight && colToLoad >= 0 && colToLoad < inputWidth)
            {
                tile[(ty + i) * (blockDim.x * stride) + (tx + j)] = input[batchIndex * inputWidth * inputHeight + rowToLoad * inputWidth + colToLoad];
            }
            else
            {
                tile[(ty + i) * (blockDim.x * stride) + (tx + j)] = 0.0f;
            }
        }
    }
    __syncthreads(); // 同步操作，等所有线程所需数据都加载进共享内存

    if (batchIndex < batchSize && row < outputHeight && col < outputWidth)
    {
        float sum = 0.0f;
        for (int i = 0; i < filterHeight; ++i)
        {
            for (int j = 0; j < filterWidth; ++j)
            {
                sum += tile[(ty + i) * (blockDim.x * stride) + (tx + j)] * filter[i * filterWidth + j];
            }
        }
        output[batchIndex * outputWidth * outputHeight + row * outputWidth + col] = sum;
    }
}

// 全连接层的矩阵乘法
__global__ void matrixMultiply(float *input, float *weight, float *output, int numinRows, int inputNeurons, int outputNeurons, int batchSize)
{

    int batchIndex = blockIdx.z;                     // 批次索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 确定当前线程对应的输出行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 确定当前线程对应的输出列

    if (batchIndex < batchSize && row < numinRows && col < outputNeurons)
    {
        float sum = 0.0f;
        for (int k = 0; k < inputNeurons; k++)
        {
            sum += input[batchIndex * numinRows * inputNeurons + row * inputNeurons + k] * weight[k * outputNeurons + col]; // 行列式相乘，每个元素相乘求和
        }
        output[batchIndex * numinRows * outputNeurons + row * outputNeurons + col] = sum; // 赋值给输出矩阵
    }
}

// 激活函数
__global__ void relu(float *input, int sizePerBatch, int batchSize)
{

    int batchIndex = blockIdx.z; // 批次索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIndex < batchSize && index < sizePerBatch)
    {
        int globalIndex = batchIndex * sizePerBatch + index; // 计算全局索引
        input[globalIndex] = max(0.0f, input[globalIndex]);  // 小于0取0，大于0线性增长
    }
}

// 最大池化
__global__ void maxpool(float *input, float *output, int inputWidth, int inputHeight, int poolSize, int batchSize)
{

    int batchIndex = blockIdx.z; // 批次索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    if (batchIndex < batchSize && row < inputHeight / poolSize && col < inputWidth / poolSize)
    {
        float maxVal = -FLT_MAX; // 初始化为最小值
        for (int i = 0; i < poolSize; ++i)
        {
            for (int j = 0; j < poolSize; ++j)
            {
                int rowIndex = row * poolSize + i;
                int colIndex = col * poolSize + j;
                if (rowIndex < inputHeight && colIndex < inputWidth)
                {
                    int index = batchIndex * inputHeight * inputWidth + rowIndex * inputWidth + colIndex;
                    maxVal = fmaxf(maxVal, input[index]); // 找一个池化框内最大的值作为下一层节点值
                }
            }
        }
        int outIndex = batchIndex * (inputHeight / poolSize) * (inputWidth / poolSize) + row * (inputWidth / poolSize) + col;
        output[outIndex] = maxVal;
    }
}

// softmax归一化
__global__ void softmax(float *input, float *output, int length, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIndex = blockIdx.z;

    int i = y * gridDim.x * blockDim.x + x; // 计算在批次中的索引

    if (batchIndex < batchSize && i < length) {
        // 计算当前样本的起始索引
        int startIndex = batchIndex * length;

        // 第一步：找到最大值，用于数值稳定性
        float maxVal = input[startIndex]; // 初始化为第一个元素的值
        for (int j = 1; j < length; ++j)
        {
            if (input[startIndex + j] > maxVal)
            {
                maxVal = input[startIndex + j]; // 更新最大值
            }
        }

        // 第二步：计算指数和
        float sum = 0.0f;
        for (int j = 0; j < length; ++j)
        {
            sum += exp(input[startIndex + j] - maxVal); // 计算所有元素的指数和
        }

        // 第三步：计算softmax
        output[startIndex + i] = exp(input[startIndex + i] - maxVal) / sum; // 计算每个元素的softmax值
    }
}

// 均方误差损失函数
__global__ void mseloss(float *predictions, int *targets, float *loss, int sizePerBatch, int batchSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y;

    if (batchIndex < batchSize && i < sizePerBatch)
    {
        int globalIndex = batchIndex * sizePerBatch + i;              // 计算全局索引
        float diff = predictions[globalIndex] - targets[globalIndex]; // 计算误差
        loss[globalIndex] = diff * diff;                              // 损失值为误差平方
    }
}

// 随机初始化权重矩阵
__global__ void initWeights(float *weights, int size, float stdDev, unsigned long long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局索引，确定当前线程应处理的元素

    if (index < size) { 
        curandState state; // 声明一个随机数生成器的状态
        curand_init(seed, index, 0, &state); // 初始化随机数生成器，为每个线程设置一个独立的序列

        weights[index] = curand_normal(&state) * stdDev; // 生成一个正态分布的随机数，并乘以标准差
    }
}

// 随机初始化过滤器
__global__ void initConvFilters(float *filters, int filterSize, int numFilters, float stdDev, unsigned long long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int filterArea = filterSize * filterSize; // 计算单个过滤器的面积

    if (index < numFilters * filterArea) { // 确保线程处理的索引在过滤器数组的有效范围内
        curandState state; // 随机数生成器的状态
        curand_init(seed, index, 0, &state); // 初始化随机数生成器

        filters[index] = curand_normal(&state) * stdDev; // 生成正态分布的随机数并乘以标准差（标准差和上一个不一样，这里用的是He初始化）
    }
}

// 以上代码是前向传播所需函数

// 以下是反向传播部分代码，未实现！
/*
__global__ void backpropMSE(float *predictions, float *targets, float *deltas, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        deltas[index] = predictions[index] - targets[index];
    }
}

__global__ void backpropFC(float *deltas, float *weights, float *prevDeltas, int numRows, int numCols)
{
    // 计算全连接层的反向传播
    // ...
}

__global__ void updateWeightsFC(float *weights, float *weightDeltas, float learningRate, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        weights[index] -= learningRate * weightDeltas[index];
    }
}

__global__ void backpropConv(float *deltas, float *filters, float *prevDeltas, int inputWidth, int inputHeight, int filterWidth, int filterHeight)
{
    // 计算卷积层的反向传播
    // ...
}

__global__ void updateWeightsConv(float *filters, float *filterDeltas, float learningRate, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        filters[index] -= learningRate * filterDeltas[index];
    }
}
*/
// 以上是未实现的反向传播代码，未实现！

// 主函数
int main()
{ // 为输入、输出、权重等分配GPU内存
    float *input, *output, *filters, *pooledOutput, *fcOutput, *finalOutput;
    cudaMalloc(&input, inputSize * inputSize * batchSize * sizeof(float));
    cudaMalloc(&output, convOutputSize * convOutputSize * numFilters * batchSize * sizeof(float));
    cudaMalloc(&filters, filterSize * filterSize * numFilters * sizeof(float)); // 过滤器数量通常不依赖于批次大小
    cudaMalloc(&pooledOutput, pooledOutputSize * pooledOutputSize * numFilters * batchSize * sizeof(float));
    cudaMalloc(&fcOutput, fcOutputSize * batchSize * sizeof(float));
    cudaMalloc(&finalOutput, finalOutputSize * batchSize * sizeof(float));

    // ... 其他必要的内存分配和初始化

    // 在主机内存中分配和初始化输入数据
    float *hostInputData = (float *)malloc(inputSize * inputSize * batchSize * sizeof(float));
    // 填充 hostInputData，例如加载图像数据到这个数组中

    // 在 GPU 上分配输入数据内存
    float *deviceInput;
    cudaMalloc(&deviceInput, inputSize * inputSize * batchSize * sizeof(float));

    // 将输入数据从主机复制到设备
    cudaMemcpy(deviceInput, hostInputData, inputSize * inputSize * batchSize * sizeof(float), cudaMemcpyHostToDevice);

    // 在主机内存中分配和初始化目标标签数据
    int *hostTargetLabels = (int *)malloc(batchSize * sizeof(int));
    // 填充 hostTargetLabels，例如加载目标标签数据到这个数组中

    // 在 GPU 上分配目标标签数据内存
    int *deviceTargetLabels;
    cudaMalloc(&deviceTargetLabels, batchSize * sizeof(int));

    // 将目标标签数据从主机复制到设备
    cudaMemcpy(deviceTargetLabels, hostTargetLabels, batchSize * sizeof(int), cudaMemcpyHostToDevice);

    // 在GPU上为Softmax输出分配内存
    float *softmaxOutput;
    cudaMalloc(&softmaxOutput, finalOutputSize * batchSize * sizeof(float));

    // ##！！这里需要目标数据，以及目标标签的输入！！##//

    // 在GPU上为损失分配内存
    float *loss;
    cudaMalloc(&loss, batchSize * sizeof(float));

    // 设置合适的网格和块尺寸
    dim3 gridSize(16, 16, batchSize); // 假设的网格大小
    dim3 blockSize(16, 16);           // 假设的块大小

    // ... 后续操作

    // 初始化过滤器和权重
    // 假设的全连接层输入大小
    const int fcInputSize = pooledOutputSize * pooledOutputSize * numFilters; // 目前还没算出全连接层大小，声明一个稍大的空间

    // 为全连接层权重分配GPU内存
    float *fcWeights;
    cudaMalloc(&fcWeights, fcInputSize * fcOutputSize * sizeof(float));

    // 初始化全连接层权重
    float stdDev = 0.01;           // 假设的标准差
    unsigned long long seed = 123; // 随机种子

    int totalWeights = fcInputSize * fcOutputSize;
    dim3 gridSizeFc((totalWeights + 255) / 256); // 以256为块大小计算网格尺寸
    dim3 blockSizeFc(256);                       // 256线程的块

    initWeights<<<gridSizeFc, blockSizeFc>>>(fcWeights, totalWeights, stdDev, seed);

    // 初始化卷积层过滤器
    float stdDevConv = sqrt(2.0f / (filterSize * filterSize)); // He初始化标准差
    unsigned long long seed = 456;                             // 随机种子

    int totalFilterSize = numFilters * filterSize * filterSize;
    dim3 gridSizeConv((totalFilterSize + 255) / 256);
    dim3 blockSizeConv(256);

    initConvFilters<<<gridSizeConv, blockSizeConv>>>(filters, filterSize, numFilters, stdDevConv, seed);

    // const int convOutputSize = (inputSize - filterSize + 1); // 卷积层输出的尺寸
    // const int pooledOutputSize = convOutputSize / poolSize;  // 池化层输出的尺寸
    // 移到开头了

    size_t convOutputMemSize = convOutputSize * convOutputSize * numFilters * sizeof(float);       // 卷积层输出的内存大小
    size_t pooledOutputMemSize = pooledOutputSize * pooledOutputSize * numFilters * sizeof(float); // 池化层输出的内存大小

    float *inputBuffer, *outputBuffer; // 初始化两个缓冲区
    cudaMalloc(&inputBuffer, convOutputMemSize);
    cudaMalloc(&outputBuffer, pooledOutputMemSize);

    // 初始化 inputBuffer
    cudaMemcpy(inputBuffer, input, convOutputMemSize, cudaMemcpyDeviceToDevice);

    // ################ 卷积层 ################

    int inputSizeConv = inputSize;

    for (int i = 0; i < ConvTime; ++i)
    { // 假设有 5 个卷积层

        // 卷积层
        convolve<<<gridSize, blockSize>>>(inputBuffer, outputBuffer, filters, inputSizeConv, inputSizeConv, filterSize, filterSize, batchSize);

        inputSizeConv = (inputSizeConv - filterSize + 2 * padding) / stride + 1; // 更新卷积后的尺寸

        // 计算卷积层的 sizePerBatch
        int sizePerBatchConv = inputSizeConv * inputSizeConv * numFilters;
        relu<<<gridSize, blockSize>>>(inputBuffer, sizePerBatchConv, batchSize);

        // 池化层
        maxpool<<gridSize, blockSize>>>(inputBuffer, outputBuffer, inputSizeConv, inputSizeConv, poolSize, batchSize);
        inputSizeConv = inputSizeConv / poolSize; // 更新池化后的尺寸

        // 交换 inputBuffer 和 outputBuffer，以更新输入为下一层的输出
        float *temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;
    }

    // ################ 全连接层 ################
    int fcOutputSize[4] = {0, 128, 64, 0};           // 每层神经元个数
    fcOutputSize[3] = typenum;                       // 最后一层全链接神经元数为需要划分的种类
    fcOutputSize[0] = inputSizeConv * inputSizeConv; // 将2维的特征平铺成一维用于全连接操作
    for (int i = 1; i <= FcTime - 1; ++i)
    { // 假设有 2 个全连接层
        matrixMultiply<<<gridSize, blockSize>>>(inputBuffer, fcWeights, fcOutput, batchSize, fcOutputSize[i - 1], fcOutputSize[i], batchSize);

        // 更新 sizePerBatch 为全连接层的输出大小
        int sizePerBatchFc = fcOutputSize[i];
        relu<<<gridSize, blockSize>>>(fcOutput, sizePerBatchFc, batchSize);

        // 为下一层更新输入缓冲区
        inputBuffer = fcOutput; // 确保 fcOutput 被正确更新
    }

    // 最后一个全连接层
    matrixMultiply<<<gridSize, blockSize>>>(inputBuffer, fcWeights, fcOutput, batchSize, fcOutputSize[2], fcOutputSize[3], batchSize);
    // 为下一层更新输入缓冲区
    inputBuffer = fcOutput; // 确保 fcOutput 被正确更新

    int length = fcOutputSize[3]; // softmax输入特征个数为最后一层全链接神经元数
    // Softmax归一化
    softmax<<<gridSize, blockSize>>>(inputBuffer, output, length, batchSize);

    // 最后一层输出为预测结果

    // 计算损失
    int sizePerBatch = finalOutputSize;
    mseloss<<<gridSize, blockSize>>>(output, deviceTargetLabels, loss, sizePerBatch, batchSize);
    // 以上是向传播代码

    // 以下是主函数中的反向传播部分，未实现！

    // 计算损失函数的梯度
    // backpropMSE<<<gridSize, blockSize>>>(output, deviceTargetLabels, deltas, sizePerBatch);

    // 反向传播到全连接层
    // backpropFC<<<gridSize, blockSize>>>(deltas, fcWeights, prevDeltas, numRowsFC, numColsFC);

    // 更新全连接层的权重
    // updateWeightsFC<<<gridSize, blockSize>>>(fcWeights, weightDeltas, learningRate, sizeWeightsFC);

    // 反向传播到卷积层
    // backpropConv<<<gridSize, blockSize>>>(prevDeltas, filters, prevPrevDeltas, inputWidthConv, inputHeightConv, filterWidth, filterHeight);

    // 更新卷积层的权重
    // updateWeightsConv<<<gridSize, blockSize>>>(filters, filterDeltas, learningRate, sizeFilters);

    // ... 其他反向传播操作 ...

    // 后续操作，例如从GPU复制数据回CPU，释放内存等
    // 释放内存
    cudaFree(input);
    cudaFree(output);
    cudaFree(filters);
    cudaFree(pooledOutput);
    cudaFree(fcOutput);
    cudaFree(finalOutput);

    cudaFree(softmaxOutput);
    cudaFree(loss);

    free(hostInputData);
    cudaFree(deviceInput);

    cudaFree(fcWeights);

    free(hostTargetLabels);
    cudaFree(deviceTargetLabels);

    cudaFree(inputBuffer);
    cudaFree(outputBuffer);

    return 0;
}