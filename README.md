# Forward Propagation of CNN Using CUDA

## Overview
This project involves implementing the forward propagation process of a Convolutional Neural Network (CNN) using CUDA C++. The focus is on foundational components like convolution, activation, pooling, and fully connected layers.

---

## Environment
- **Development**: CUDA C++ files written on macOS using VS Code.
- **Execution**: Tested via SSH on a GPGPU server provided by the instructor.

---

## Algorithm Details
### Implemented Components:
1. **Convolution Kernel**: Extracts features using sliding window operations with 3x3 filters.
2. **ReLU Activation**: Sets negative values to 0 to enhance non-linearity.
3. **MaxPooling**: Reduces dimensions using 2x2 pooling windows.
4. **Fully Connected Layers**: Combines features using matrix multiplication.
5. **Softmax Normalization**: Converts outputs into probabilities.
6. **MSE Loss Function**: Computes errors between predictions and true values.
7. **Random Initialization**: Initializes weights and filters using normal distributions.

### Network Architecture:
- **Layers**: 8 total (5 convolutional, 3 fully connected).
- **Structure**:  
  Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → … → FC → ReLU → FC → ReLU → FC → Softmax
- **Convolution Parameters**:
  - Filter size: 3x3
  - Filters: 16
  - Padding: 0
  - Stride: 1
- **Fully Connected Parameters**:
  - Layer 1: 128 neurons
  - Layer 2: 64 neurons
  - Layer 3 (Output): 10 neurons

---

## Workflow
1. Allocate GPU memory for inputs, weights, and outputs.
2. Copy data to GPU memory.
3. Perform forward propagation:
   - Convolution → Activation → Pooling → Fully Connected → Softmax
4. Compute loss using MSE.
5. Free allocated GPU memory after execution.

---

## Key Observations
### Performance Analysis:
1. **Parallel Processing**:
   - Leveraged CUDA to parallelize convolution, activation, pooling, and fully connected operations.
   - GPU-accelerated convolution outperformed CPU by up to 1000x.

2. **Batch Processing**:
   - Batch size optimizations improved efficiency and adaptability.
   - Misconfigured batch sizes can lead to crashes or slow training.

3. **Bottlenecks**:
   - Convolution operations were the most computationally expensive.
   - Shared memory and tiling were applied to optimize convolution efficiency.

4. **Quantitative Insights**:
   - First convolution layer: ~13M calculations.
   - Fully connected layers: Total ~27k calculations.
   - Convolution layers dominate computational workload.

5. **Scalability**:
   - Flexible parameterization allows easy adaptation for larger or smaller networks.
   - Batch size adjustments are crucial for different hardware setups.

6. **Memory Access**:
   - Non-contiguous memory access in convolution was addressed with shared memory optimizations.
   - Data transfer (host ↔ device) significantly impacts performance.

---

## Limitations and Future Work
- **Backward Propagation**: Not implemented but includes:
  - Gradient calculation for loss.
  - Weight updates for convolutional and fully connected layers.
- **Performance Bottlenecks**: Addressed partially with shared memory optimizations.

---

## Experiment Results
- Developed a functioning forward propagation pipeline for CNNs.
- Demonstrated efficient GPU utilization and scalability across varying network parameters.

---

## Conclusion
This project illustrates the potential of GPU-accelerated neural network computations, emphasizing convolutional operations. While forward propagation was completed successfully, extending to backpropagation remains a goal for further research.
