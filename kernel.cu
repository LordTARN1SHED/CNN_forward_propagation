#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

//###########################//

//######### 前向传播 #########//

//###########################//

//卷积层
__global__ void convolve(float *input, float *output, float *filter, int inputWidth, int inputHeight, int filterWidth, int filterHeight) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < inputHeight && col < inputWidth) {
        for (int i = 0; i < filterHeight; ++i) {
            for (int j = 0; j < filterWidth; ++j) {
                int rowOffset = row + i - filterHeight / 2;
                int colOffset = col + j - filterWidth / 2;
                if (rowOffset >= 0 && rowOffset < inputHeight && colOffset >= 0 && colOffset < inputWidth) {
                    sum += filter[i * filterWidth + j] * input[rowOffset * inputWidth + colOffset];
                }
            }
        }
        output[row * inputWidth + col] = sum;
    }
}

//全连接层
__global__ void matrixMultiply(float *input, float *weight, float *output, int numinRows, int numinColumns, int numwColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 确定当前线程对应的输出行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 确定当前线程对应的输出列

    // 确保当前线程处理的是有效的输出位置
    if (row < numinRows && col < numwColumns) {
        float sum = 0.0f;
        // 计算矩阵A的一行与矩阵B的一列的点积
        for (int k = 0; k < numinColumns; k++) {
            sum += input[row * numinColumns + k] * weight[k * numwColumns + col];
        }
        // 将点积结果存储在输出矩阵C的相应位置
        output[row * numwColumns + col] = sum;
    }
}


//激活函数
__global__ void relu(float *input, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        input[index] = max(0.0f, input[index]);
    }
}

//最大池化
__global__ void maxpool(float *input, float *output, int inputWidth, int inputHeight, int poolSize) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float maxVal = -FLT_MAX;
    if (row < inputHeight / poolSize && col < inputWidth / poolSize) {
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int rowIndex = row * poolSize + i;
                int colIndex = col * poolSize + j;
                if (rowIndex < inputHeight && colIndex < inputWidth) {
                    maxVal = fmaxf(maxVal, input[rowIndex * inputWidth + colIndex]);
                }
            }
        }
        output[row * (inputWidth / poolSize) + col] = maxVal;
    }
}

//softmax归一化
__global__ void softmax(float *input, float *output, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        // Step 1: Find the max value for numerical stability
        float maxVal = input[0];
        for (int j = 1; j < length; ++j) {
            if (input[j] > maxVal) {
                maxVal = input[j];
            }
        }

        // Step 2: Compute the exponential sum
        float sum = 0.0f;
        for (int j = 0; j < length; ++j) {
            sum += exp(input[j] - maxVal);
        }

        // Step 3: Compute the softmax
        output[i] = exp(input[i] - maxVal) / sum;
    }
}

//均方误差损失函数
__global__ void mse_loss(float *predictions, float *targets, float *loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = predictions[i] - targets[i];
        loss[i] = diff * diff;
    }
}

//随机初始化权重矩阵/过滤器
__global__ void initWeights(float *weights, int size, float stdDev, unsigned long long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        curandState state;
        curand_init(seed, index, 0, &state);
        weights[index] = curand_normal(&state) * stdDev;
    }
}

//###########################//

//######### 反向传播 #########//

//###########################//


// 假设这个函数计算了最后一层的误差梯度
__global__ void computeLastLayerGradients(float *lastLayerDeltas, float *outputs, float *targets, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        lastLayerDeltas[i] = (outputs[i] - targets[i]); // 对于MSE损失函数
    }
}

// 更新权重
__global__ void updateWeights(float *weights, float *weightGradients, float learningRate, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        weights[i] -= learningRate * weightGradients[i];
    }
}

// 计算权重梯度（简化版）
__global__ void computeWeightGradients(float *weightGradients, float *lastLayerDeltas, float *previousLayerOutputs, int numWeights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numWeights) {
        // 假设是一个简单的全连接层
        weightGradients[i] = lastLayerDeltas[i / numOutputs] * previousLayerOutputs[i % numInputs];
    }
}



__global__ void update_weights(float *weights, float *gradients, int size, float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        weights[index] -= learning_rate * gradients[index];
    }
}

__global__ void mse_loss_backward(float *d_predictions, float *d_targets, float *d_loss_grad, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // 损失函数的梯度：2*(预测值 - 目标值)
        d_loss_grad[index] = 2.0f * (d_predictions[index] - d_targets[index]);
    }
}

__global__ void relu_backward(float *d_input, float *d_output_grad, float *d_input_grad, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        d_input_grad[index] = (d_input[index] > 0) ? d_output_grad[index] : 0.0f;
    }
}

__global__ void conv_backward(float *d_input, float *d_output_grad, float *d_weights_grad, int inputWidth, int filterWidth) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int outputWidth = inputWidth - filterWidth + 1;

    if (ix < filterWidth && iy < filterWidth) {
        float sum = 0;
        for (int ox = 0; ox < outputWidth; ++ox) {
            for (int oy = 0; oy < outputWidth; ++oy) {
                sum += d_input[(oy + iy) * inputWidth + (ox + ix)] * d_output_grad[oy * outputWidth + ox];
            }
        }
        d_weights_grad[iy * filterWidth + ix] = sum;
    }
}

__global__ void update_weights(float *weights, float *gradients, int size, float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        weights[index] -= learning_rate * gradients[index];
    }
}

