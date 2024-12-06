# MNIST CNN Classifier with CI/CD Pipeline

A CNN implementation for MNIST digit classification with automated testing and CI/CD pipeline.

## Model Architecture Summary
==================================================

### Layer Details:
1. **First Block - Initial Feature Extraction**
   - Conv1: 1 → 16 channels (3x3)
   - BatchNorm, ReLU, Dropout(0.05)
   - Output: 28x28 → 26x26
   - Parameters: 160

2. **Additional Feature Extraction**
   - Conv11: 16 → 16 channels (3x3)
   - BatchNorm, ReLU, Dropout(0.05)
   - Output: 26x26 → 24x24
   - Parameters: 2,320

3. **Second Block - Feature Processing**
   - Conv2: 16 → 32 channels (3x3)
   - BatchNorm, ReLU, Dropout(0.05)
   - MaxPool2d(2,2)
   - Output: 24x24 → 12x12
   - Parameters: 4,640

4. **Third Block - Channel Reduction**
   - Conv3: 32 → 16 channels (1x1)
   - BatchNorm, ReLU, Dropout(0.05)
   - Output: 12x12 → 12x12
   - Parameters: 528

5. **Fourth Block - Feature Processing**
   - Conv4: 16 → 32 channels (3x3)
   - BatchNorm, ReLU, Dropout(0.05)
   - Output: 12x12 → 10x10
   - Parameters: 4,640

6. **Fifth Block**
   - Conv5: 32 → 32 channels (3x3)
   - BatchNorm, ReLU, Dropout(0.05)
   - Output: 10x10 → 8x8
   - Parameters: 9,248

7. **Final Classification**
   - Conv6: 32 → 10 channels (1x1)
   - Global Average Pooling
   - Output: 8x8 → 1x1
   - Parameters: 330

Total Parameters: 22,090

## Test Cases
==================================================

### 1. Parameter Count Test
### 2. Batch Normalization Test
### 3. Dropout Test
### 4. Global Average Pooling Test

## Test Results
- ✅ Parameter Count: Passed (22,090 < 50,000)
- ✅ BatchNorm Usage: Passed (Used in all conv blocks)
- ✅ Dropout Usage: Passed (p=0.05 throughout)
- ✅ GAP Usage: Passed (No FC layers used)

## Training Configuration
1. **Optimizer**: Adam
   - Learning rate: 0.001
   - Weight decay: 5e-4

2. **Scheduler**: OneCycleLR
   - Max LR: 0.005
   - Epochs: 20
   - Warmup: 30%

3. **Data Augmentation**:
   - Random Rotation (3°)
   - Random Affine (shear=3)
   - Normalization (mean=0.1307, std=0.3081)

## Results
- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Test Accuracy: ~99%