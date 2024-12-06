import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import ssl
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First block - Initial feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)     # 28x28 -> 26x26 (32 channels)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(0.05)

         # First block - Initial feature extraction
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3)     # 28x28 -> 26x26 (32 channels)
        self.bn11 = nn.BatchNorm2d(16)
        self.dropout11 = nn.Dropout2d(0.05)
        
        # Second block - Feature processing
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)    # 26x26 -> 24x24 (64 channels)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(0.05)
        self.pool1 = nn.MaxPool2d(2, 2)                  # 24x24 -> 12x12
        
        # Third block - Feature extraction with 1x1
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)    # 12x12 -> 12x12 (32 channels)
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(0.05)
        
        # Fourth block - Final processing
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)    # 12x12 -> 10x10 (64 channels)
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(0.05)

         # Fifth block - 
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3)     # 28x28 -> 26x26 (32 channels)
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout2d(0.05)
        
        
        # Final classification block
        self.conv6 = nn.Conv2d(16, 10, kernel_size=1)    # 10x10 -> 10x10 (10 channels)
        self.gap = nn.AdaptiveAvgPool2d(1)               # Global Average Pooling
        self.relu = nn.ReLU()

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.dropout11(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.pool1(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # Fifth block
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout5(x)
        
        # Final classification
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x

def print_model_summary(model):
    print("\n" + "="*50)
    print("Model Architecture Summary")
    print("="*50)
    
    print("\nLayer Details and Parameter Count:")
    print("-"*50)
    total_params = 0
    total_conv_params = 0
    total_bn_params = 0
    
    # First Block - Initial Feature Extraction
    conv1_params = sum(p.numel() for p in model.conv1.parameters())
    bn1_params = sum(p.numel() for p in model.bn1.parameters())
    print("1. First Block - Initial Feature Extraction")
    print(f"   - Conv1: 1 → 16 channels (3x3)")
    print(f"   - Parameters: {conv1_params:,} ({conv1_params-16} weights + 16 bias)")
    print(f"   - BatchNorm1: {bn1_params} parameters ({bn1_params//2} weights + {bn1_params//2} bias)")
    total_conv_params += conv1_params
    total_bn_params += bn1_params
    
    # Additional Feature Extraction
    conv11_params = sum(p.numel() for p in model.conv11.parameters())
    bn11_params = sum(p.numel() for p in model.bn11.parameters())
    print("\n2. Additional Feature Extraction")
    print(f"   - Conv11: 16 → 16 channels (3x3)")
    print(f"   - Parameters: {conv11_params:,} ({conv11_params-16} weights + 16 bias)")
    print(f"   - BatchNorm11: {bn11_params} parameters ({bn11_params//2} weights + {bn11_params//2} bias)")
    total_conv_params += conv11_params
    total_bn_params += bn11_params
    
    # Second Block - Feature Processing
    conv2_params = sum(p.numel() for p in model.conv2.parameters())
    bn2_params = sum(p.numel() for p in model.bn2.parameters())
    print("\n3. Second Block - Feature Processing")
    print(f"   - Conv2: 16 → 32 channels (3x3)")
    print(f"   - Parameters: {conv2_params:,} ({conv2_params-32} weights + 32 bias)")
    print(f"   - BatchNorm2: {bn2_params} parameters ({bn2_params//2} weights + {bn2_params//2} bias)")
    print(f"   - MaxPool2d: 0 parameters")
    total_conv_params += conv2_params
    total_bn_params += bn2_params
    
    # Third Block - Channel Reduction
    conv3_params = sum(p.numel() for p in model.conv3.parameters())
    bn3_params = sum(p.numel() for p in model.bn3.parameters())
    print("\n4. Third Block - Channel Reduction")
    print(f"   - Conv3: 32 → 16 channels (1x1)")
    print(f"   - Parameters: {conv3_params:,} ({conv3_params-16} weights + 16 bias)")
    print(f"   - BatchNorm3: {bn3_params} parameters ({bn3_params//2} weights + {bn3_params//2} bias)")
    total_conv_params += conv3_params
    total_bn_params += bn3_params
    
    # Fourth Block - Feature Processing
    conv4_params = sum(p.numel() for p in model.conv4.parameters())
    bn4_params = sum(p.numel() for p in model.bn4.parameters())
    print("\n5. Fourth Block - Feature Processing")
    print(f"   - Conv4: 16 → 32 channels (3x3)")
    print(f"   - Parameters: {conv4_params:,} ({conv4_params-32} weights + 32 bias)")
    print(f"   - BatchNorm4: {bn4_params} parameters ({bn4_params//2} weights + {bn4_params//2} bias)")
    total_conv_params += conv4_params
    total_bn_params += bn4_params
    
    # Fifth Block
    conv5_params = sum(p.numel() for p in model.conv5.parameters())
    bn5_params = sum(p.numel() for p in model.bn5.parameters())
    print("\n6. Fifth Block")
    print(f"   - Conv5: 32 → 16 channels (3x3)")
    print(f"   - Parameters: {conv5_params:,} ({conv5_params-16} weights + 16 bias)")
    print(f"   - BatchNorm5: {bn5_params} parameters ({bn5_params//2} weights + {bn5_params//2} bias)")
    total_conv_params += conv5_params
    total_bn_params += bn5_params
    
    # Final Classification
    conv6_params = sum(p.numel() for p in model.conv6.parameters())
    print("\n7. Final Classification")
    print(f"   - Conv6: 16 → 10 channels (1x1)")
    print(f"   - Parameters: {conv6_params:,} ({conv6_params-10} weights + 10 bias)")
    print(f"   - Global Average Pooling: 0 parameters")
    total_conv_params += conv6_params
    
    total_params = total_conv_params + total_bn_params
    
    print("\nParameter Summary:")
    print(f"- Convolutional Layers: {total_conv_params:,} parameters")
    print(f"- BatchNorm Layers: {total_bn_params:,} parameters")
    print(f"- Total Parameters: {total_params:,}")
    
    print("\nArchitecture Features:")
    print("- Uses both 3x3 and 1x1 convolutions")
    print("- BatchNorm after each conv layer")
    print("- Dropout (p=0.05) for regularization")
    print("- Global Average Pooling instead of FC")
    print("- Multiple channel reduction points")
    
    print("\n" + "="*50)

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def analyze_model_training(model, optimizer, data, target):
    """Simplified analysis of model training"""
    
    criterion = nn.CrossEntropyLoss()
    
    # 1. Check Feature Learning
    with torch.no_grad():
        # Get activations after each layer
        x = data
        
        # First layer
        x1 = model.conv1(x)
        x1 = model.bn1(x1)
        x1 = model.relu(x1)
        
        # Check if first layer is learning
        first_layer_stats = {
            'mean_activation': x1.mean().item(),
            'dead_neurons': (x1 == 0).float().mean().item() * 100
        }
        
        # Final layer
        output = model(data)
        predictions = torch.argmax(output, dim=1)
        correct = (predictions == target).float().mean().item() * 100
    
    # 2. Check Gradients
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Get gradients from first and last conv layers
    grad_stats = {
        'first_layer': {
            'mean': model.conv1.weight.grad.mean().item(),
            'max': model.conv1.weight.grad.max().item()
        },
        'last_layer': {
            'mean': model.conv5.weight.grad.mean().item(),
            'max': model.conv5.weight.grad.max().item()
        }
    }
    
    print("\n=== Model Analysis ===")
    print("\n1. Feature Learning:")
    print(f"- First layer mean activation: {first_layer_stats['mean_activation']:.4f}")
    print(f"- Dead neurons: {first_layer_stats['dead_neurons']:.1f}%")
    print(f"- Batch accuracy: {correct:.1f}%")
    
    print("\n2. Gradient Health:")
    print("First Layer Gradients:")
    print(f"- Mean: {grad_stats['first_layer']['mean']:.6f}")
    print(f"- Max: {grad_stats['first_layer']['max']:.6f}")
    print("Last Layer Gradients:")
    print(f"- Mean: {grad_stats['last_layer']['mean']:.6f}")
    print(f"- Max: {grad_stats['last_layer']['max']:.6f}")
    
    print("\n3. Training Status:")
    print(f"- Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"- Current loss: {loss.item():.4f}")
    
    # Provide insights
    print("\n=== Analysis Insights ===")
    if first_layer_stats['dead_neurons'] > 50:
        print("WARNING: Too many dead neurons - consider reducing learning rate")
    
    if abs(grad_stats['first_layer']['mean']) < 1e-7:
        print("WARNING: Very small gradients - might have vanishing gradient problem")
    
    if abs(grad_stats['first_layer']['max']) > 1:
        print("WARNING: Large gradients - might have exploding gradient problem")
    
    return {
        'feature_stats': first_layer_stats,
        'grad_stats': grad_stats,
        'loss': loss.item(),
        'accuracy': correct,
        'lr_impact': optimizer.param_groups[0]['lr']
    }

def save_augmented_samples(train_loader, num_samples=10):
    """Save augmented sample images"""
    
    # Create directory if it doesn't exist
    save_dir = 'augmented_samples'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get a batch of images
    images, labels = next(iter(train_loader))
    
    # Save the first num_samples images
    for i in range(num_samples):
        img = images[i].squeeze()  # Remove channel dimension for MNIST
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
        plt.savefig(f'{save_dir}/augmented_sample_{i+1}.png')
        plt.close()
    
    print(f"Saved {num_samples} augmented samples to '{save_dir}' directory")

def save_model(model, accuracy, epoch):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.2f}".replace(".", "p")
    filename = f"mnist_model_e{epoch}_{timestamp}_acc{accuracy_str}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename

def train_model(num_epochs=20, target_accuracy=99.5):
    print("\n" + "="*50)
    print(" Starting Training Process")
    print(f" Target Accuracy: {target_accuracy}%")
    print("="*50)
    
    # Initialize model and print summary
    model = SimpleCNN()
    print_model_summary(model)
    
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),  # First convert to tensor
    #     transforms.ToPILImage(),  # Convert tensor to PIL Image for augmentations
    #     transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),  # Convert back to tensor
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(3),
        transforms.RandomAffine(
            degrees=0,
            # translate=(0.05, 0.05),
            # scale=(0.95, 1.05),
            shear=3
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32,         # Reduce from 128 to 64
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Save augmented samples
    save_augmented_samples(train_loader)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,              # Reduce from 0.01 to 0.001
        betas=(0.9, 0.999),    
        eps=1e-8,              
        weight_decay=5e-4      # Increase weight decay slightly
    )
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.005,          # Reduce from 0.01 to 0.005
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,         # Longer warmup
    #     div_factor=25,         # Gentler learning rate curve
    #     final_div_factor=1e4
    # )
    
    # Training loop
    best_test_acc = 0
    history = {
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}]")
        print("-"*50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{num_epochs}',
            ncols=80,  # Fixed width
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
        )
        
        for data, target in pbar:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar with cleaner metrics
            train_acc = 100 * correct_train / total_train
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{train_acc:.1f}%'
            }, refresh=True)
        
        # Calculate accuracies
        train_acc = 100 * correct_train / total_train
        val_acc = evaluate_model(model, val_loader)
        test_acc = evaluate_model(model, test_loader)
        
        # Store metrics
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        # Print epoch summary
        print("\n📈 Epoch Summary:")
        print(f"├─ Training    : {train_acc:.2f}%")
        print(f"├─ Validation  : {val_acc:.2f}%")
        print(f"└─ Test        : {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_path = save_model(model, test_acc, epoch)
            print(f"\n💾 New best model saved! ({model_path})")
        
        # Early stopping if target accuracy reached
        if val_acc >= target_accuracy:
            print(f"\n🎯 Target accuracy {target_accuracy}% reached!")
            print(f"Stopping training at epoch {epoch+1}")
            break
    
    print("\n" + "="*50)
    print("🎉 Training Completed!")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print("="*50 + "\n")
    
    return model, history

if __name__ == "__main__":
    train_model(num_epochs=20, target_accuracy=99.5) 