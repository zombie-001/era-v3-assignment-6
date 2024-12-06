# MNIST CNN Classifier with CI/CD Pipeline

A CNN implementation for MNIST digit classification with automated testing and CI/CD pipeline.

## Model Architecture Summary
==================================================

### Layer Details and Parameter Count:
1. **First Block - Initial Feature Extraction**
   - Conv1: 1 → 16 channels (3x3)
   - Parameters: 160 (144 weights + 16 bias)
   - BatchNorm1: 64 parameters (32 weights + 32 bias)

2. **Additional Feature Extraction**
   - Conv11: 16 → 16 channels (3x3)
   - Parameters: 2,320 (2,304 weights + 16 bias)
   - BatchNorm11: 64 parameters (32 weights + 32 bias)

3. **Second Block - Feature Processing**
   - Conv2: 16 → 32 channels (3x3)
   - Parameters: 4,640 (4,608 weights + 32 bias)
   - BatchNorm2: 128 parameters (64 weights + 64 bias)
   - MaxPool2d: 0 parameters

4. **Third Block - Channel Reduction**
   - Conv3: 32 → 16 channels (1x1)
   - Parameters: 528 (512 weights + 16 bias)
   - BatchNorm3: 64 parameters (32 weights + 32 bias)

5. **Fourth Block - Feature Processing**
   - Conv4: 16 → 32 channels (3x3)
   - Parameters: 4,640 (4,608 weights + 32 bias)
   - BatchNorm4: 128 parameters (64 weights + 64 bias)

6. **Fifth Block**
   - Conv5: 32 → 16 channels (3x3)
   - Parameters: 4,624 (4,608 weights + 16 bias)
   - BatchNorm5: 64 parameters (32 weights + 32 bias)

7. **Final Classification**
   - Conv6: 16 → 10 channels (1x1)
   - Parameters: 170 (160 weights + 10 bias)
   - Global Average Pooling: 0 parameters

### Parameter Summary:
- Convolutional Layers: 17,082 parameters
- BatchNorm Layers: 512 parameters
- Total Parameters: 17,594

### Architecture Features:
- Uses both 3x3 and 1x1 convolutions
- BatchNorm after each conv layer
- Dropout (p=0.05) for regularization
- Global Average Pooling instead of FC
- Multiple channel reduction points

## Test Results
- ✅ Parameter Count: Passed (17,594 < 20,000)
- ✅ BatchNorm Usage: Passed (6 BatchNorm layers)
- ✅ Dropout Usage: Passed (p=0.05 throughout)
- ✅ GAP Usage: Passed (No FC layers)

## Training Results
- Best Training Accuracy: 99.32%
- Best Validation Accuracy: 99.41%
- Best Test Accuracy: 99.45%
- Convergence Time: ~15 epochs