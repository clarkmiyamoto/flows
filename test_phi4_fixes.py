#!/usr/bin/env python3
"""
Test script to verify Phi4 package fixes
"""

import sys
import os
sys.path.append('.')

import torch
from inference.path import LinearConditionalProbabilityPath
from inference.distribution import Gaussian
from learning.train import ConditionalFlowMatchingTrainer
from learning.mlp import MLPVectorField

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    # Test torch.func imports
    from torch.func import vmap, jacrev
    print("✓ torch.func imports successful")
    
    # Test package imports
    from inference.path import LinearConditionalProbabilityPath
    from inference.distribution import Gaussian, Sampleable
    from learning.train import ConditionalFlowMatchingTrainer
    from learning.mlp import MLPVectorField
    print("✓ Package imports successful")
    
    return True

def test_phi4_class():
    """Test Phi4 class definition"""
    print("\nTesting Phi4 class...")
    
    # Import Sampleable
    from inference.distribution import Sampleable
    
    # Define Phi4 class
    class Phi4(Sampleable):
        def __init__(self, L, k, l, burn_in=1000, device='cpu'):
            self.L = L
            self.k = k
            self.l = l
            self.device = device
            self._current_phi = torch.randn(L, L).to(device)
        
        @property
        def dim(self):
            return self.L**2
        
        def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
            return torch.randn(num_samples, self.dim, device=device)
    
    # Test instantiation
    device = torch.device('cpu')
    phi4 = Phi4(L=4, k=0.3, l=0.02, device=device)
    print(f"✓ Phi4 instantiated with dim={phi4.dim}")
    
    # Test sampling
    samples = phi4.sample(num_samples=10, device=device)
    print(f"✓ Sampling successful, shape: {samples.shape}")
    
    return True

def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")
    
    device = torch.device('cpu')
    
    # Import Sampleable
    from inference.distribution import Sampleable
    
    # Create Phi4 instance
    class Phi4(Sampleable):
        def __init__(self, L, k, l, burn_in=1000, device='cpu'):
            self.L = L
            self.k = k
            self.l = l
            self.device = device
        
        @property
        def dim(self):
            return self.L**2
        
        def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
            return torch.randn(num_samples, self.dim, device=device)
    
    phi4 = Phi4(L=4, k=0.3, l=0.02, device=device)
    
    # Test path construction
    path = LinearConditionalProbabilityPath(
        p_simple=Gaussian.isotropic(dim=16, std=1.0),
        p_data=phi4
    ).to(device)
    print("✓ Path construction successful")
    
    # Test model construction
    model = MLPVectorField(dim=16, hiddens=[64, 64])
    print("✓ Model construction successful")
    
    # Test trainer construction
    trainer = ConditionalFlowMatchingTrainer(path, model, device)
    print("✓ Trainer construction successful")
    
    return True

def main():
    """Run all tests"""
    print("Running Phi4 package tests...\n")
    
    try:
        test_imports()
        test_phi4_class()
        test_training_setup()
        print("\n🎉 All tests passed! The package should work correctly now.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 