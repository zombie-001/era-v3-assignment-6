import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from eva3_session_6_assignment_ import SimpleCNN

# Force CPU usage
device = torch.device('cpu')

def test_parameter_count():
    model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 50000, f"Model has {total_params:,} parameters, should be < 50,000"

def test_batch_norm_usage():
    model = SimpleCNN().to(device)
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use Batch Normalization"

def test_dropout_usage():
    model = SimpleCNN().to(device)
    has_dropout = any(isinstance(m, torch.nn.Dropout2d) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_usage():
    model = SimpleCNN().to(device)
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling" 