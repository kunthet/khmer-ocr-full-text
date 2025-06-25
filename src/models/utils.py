"""
Model utilities for summary, analysis, and debugging.

Provides tools for model inspection, parameter counting, memory usage analysis,
and model architecture visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import OrderedDict


class ModelSummary:
    """
    Model summary and analysis utility.
    
    Provides detailed information about model architecture,
    parameters, memory usage, and computational requirements.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model summary.
        
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self._summary_data = None
    
    def summary(self, 
                input_size: Tuple[int, ...] = (3, 64, 128),
                batch_size: int = 1,
                device: str = 'cpu') -> Dict[str, Any]:
        """
        Generate comprehensive model summary.
        
        Args:
            input_size: Input tensor size (C, H, W)
            batch_size: Batch size for analysis
            device: Device to run analysis on
            
        Returns:
            Dictionary with model summary information
        """
        self.model.eval()
        
        # Move model to device
        model_device = next(self.model.parameters()).device
        if str(model_device) != device:
            self.model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        
        # Hook to capture layer information
        summary_data = OrderedDict()
        hooks = []
        
        def register_hook(module, name):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary_data)
                
                m_key = f"{name}_{class_name}_{module_idx}"
                summary_data[m_key] = OrderedDict()
                summary_data[m_key]["input_shape"] = list(input[0].size()) if input else []
                summary_data[m_key]["output_shape"] = list(output.size()) if hasattr(output, 'size') else []
                summary_data[m_key]["trainable"] = any(p.requires_grad for p in module.parameters())
                
                if hasattr(output, 'size'):
                    summary_data[m_key]["nb_params"] = sum(p.numel() for p in module.parameters())
                else:
                    summary_data[m_key]["nb_params"] = 0
                    
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = register_hook(module, name)
                hooks.append(module.register_forward_hook(hook))
        
        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate summary statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Estimate memory usage
        input_size_mb = (batch_size * torch.tensor(input_size).prod() * 4) / (1024**2)
        params_size_mb = total_params * 4 / (1024**2)  # Assuming float32
        
        # Model size estimation
        total_output_size = 0
        for layer_info in summary_data.values():
            if layer_info["output_shape"]:
                output_size = torch.tensor(layer_info["output_shape"]).prod()
                total_output_size += output_size
        
        forward_backward_size_mb = total_output_size * 4 / (1024**2) * 2  # Forward + backward
        
        # Restore original device
        self.model.to(model_device)
        
        summary = {
            "model_name": self.model.__class__.__name__,
            "input_size": [batch_size] + list(input_size),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "model_size_mb": params_size_mb,
            "estimated_memory_usage": {
                "input_mb": float(input_size_mb),
                "params_mb": float(params_size_mb),
                "forward_backward_mb": float(forward_backward_size_mb),
                "total_mb": float(input_size_mb + params_size_mb + forward_backward_size_mb)
            },
            "layer_details": summary_data
        }
        
        self._summary_data = summary
        return summary
    
    def print_summary(self, 
                     input_size: Tuple[int, ...] = (3, 64, 128),
                     batch_size: int = 1):
        """
        Print formatted model summary.
        
        Args:
            input_size: Input tensor size
            batch_size: Batch size
        """
        summary = self.summary(input_size, batch_size)
        
        print("="*80)
        print(f"Model: {summary['model_name']}")
        print("="*80)
        print(f"Input size: {summary['input_size']}")
        print("-"*80)
        print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
        print("="*80)
        
        for layer_name, layer_info in summary['layer_details'].items():
            output_shape = str(layer_info['output_shape'])
            param_count = f"{layer_info['nb_params']:,}"
            print(f"{layer_name:<25} {output_shape:<25} {param_count:<15}")
        
        print("="*80)
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable params: {summary['trainable_params']:,}")
        print(f"Non-trainable params: {summary['non_trainable_params']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
        print(f"Estimated memory usage: {summary['estimated_memory_usage']['total_mb']:.2f} MB")
        print("="*80)
    
    def save_summary(self, filepath: str, **kwargs):
        """
        Save model summary to file.
        
        Args:
            filepath: Path to save summary
            **kwargs: Arguments for summary generation
        """
        if self._summary_data is None:
            self.summary(**kwargs)
        
        with open(filepath, 'w') as f:
            json.dump(self._summary_data, f, indent=2)


def count_parameters(model: nn.Module, 
                    trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Get device information
    devices = {str(p.device) for p in model.parameters()}
    
    # Get data types
    dtypes = {str(p.dtype) for p in model.parameters()}
    
    # Calculate model size
    model_size_mb = total_params * 4 / (1024**2)  # Assuming float32
    
    info = {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "devices": list(devices),
        "parameter_dtypes": list(dtypes),
        "training_mode": model.training
    }
    
    return info


def profile_model(model: nn.Module,
                 input_size: Tuple[int, ...] = (3, 64, 128),
                 batch_size: int = 1,
                 num_runs: int = 10,
                 device: str = 'cpu') -> Dict[str, float]:
    """
    Profile model performance.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        batch_size: Batch size
        num_runs: Number of runs for timing
        device: Device to run profiling on
        
    Returns:
        Performance metrics
    """
    import time
    
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Time forward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    throughput = batch_size / avg_inference_time
    
    # Memory usage (for CUDA)
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    metrics = {
        "avg_inference_time_ms": avg_inference_time * 1000,
        "throughput_samples_per_sec": throughput,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "device": device
    }
    
    return metrics


def compare_models(models: Dict[str, nn.Module],
                  input_size: Tuple[int, ...] = (3, 64, 128),
                  batch_size: int = 1) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model names and instances
        input_size: Input tensor size
        batch_size: Batch size
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for name, model in models.items():
        try:
            info = get_model_info(model)
            summary = ModelSummary(model).summary(input_size, batch_size)
            
            comparison[name] = {
                "parameters": info["total_parameters"],
                "trainable_parameters": info["trainable_parameters"],
                "model_size_mb": info["model_size_mb"],
                "estimated_memory_mb": summary["estimated_memory_usage"]["total_mb"]
            }
        except Exception as e:
            comparison[name] = {"error": str(e)}
    
    return comparison


def visualize_architecture(model: nn.Module, 
                          input_size: Tuple[int, ...] = (3, 64, 128),
                          save_path: Optional[str] = None) -> str:
    """
    Generate text-based architecture visualization.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        save_path: Path to save visualization (optional)
        
    Returns:
        Architecture visualization string
    """
    visualization = []
    visualization.append("="*60)
    visualization.append(f"Model Architecture: {model.__class__.__name__}")
    visualization.append("="*60)
    
    def add_module_info(module, prefix="", depth=0):
        indent = "  " * depth
        class_name = module.__class__.__name__
        
        # Get module parameters
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if params > 0:
            visualization.append(f"{indent}{prefix}{class_name} ({params:,} params)")
        else:
            visualization.append(f"{indent}{prefix}{class_name}")
        
        # Add children
        for name, child in module.named_children():
            add_module_info(child, f"{name}: ", depth + 1)
    
    add_module_info(model)
    
    # Add summary
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    visualization.append("="*60)
    visualization.append(f"Total Parameters: {total_params:,}")
    visualization.append(f"Trainable Parameters: {trainable_params:,}")
    visualization.append(f"Model Size: {total_params * 4 / (1024**2):.2f} MB")
    visualization.append("="*60)
    
    result = "\n".join(visualization)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(result)
    
    return result 