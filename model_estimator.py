import torch
from model import VLA_transformer

class ModelSizeEstimator:
    def __init__(self, model):
        self.model = model

    def get_parameter_count(self):
        """Calculates the total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_parameter_size_mb(self):
        """Calculates the model size in MB (assuming float32)."""
        param_count = self.get_parameter_count()
        # 4 bytes per float32
        return (param_count * 4) / (1024 ** 2)

    def estimate_memory_requirements(self, batch_size, optimizer_type='adam', mixed_precision=False):
        """
        Estimates the GPU memory required for training.
        
        Args:
            batch_size (int): The batch size used for training.
            optimizer_type (str): 'adam' or 'sgd'. Adam requires more memory for states.
            mixed_precision (bool): Whether mixed precision training is used.
            
        Returns:
            dict: Estimated memory usage in MB for different components.
        """
        param_count = self.get_parameter_count()
        dtype_size = 4  # float32
        
        # Model parameters
        model_mem = param_count * dtype_size
        
        # Gradients
        grad_mem = param_count * dtype_size
        
        # Optimizer states
        if optimizer_type.lower() == 'adam':
            # Adam keeps 2 states (momentums) per parameter
            optim_mem = param_count * dtype_size * 2
        elif optimizer_type.lower() == 'sgd':
            # SGD with momentum keeps 1 state, without momentum 0. Assuming momentum here.
            optim_mem = param_count * dtype_size
        else:
            optim_mem = 0
            
        # Activations (Rough estimate: usually proportional to model size and batch size, 
        # but highly dependent on architecture. A heuristic is often used if not tracing.)
        # This is a very rough heuristic. For Transformers/ResNets, activations can be large.
        # A safer lower bound estimate or a heuristic factor.
        # Let's assume activations take roughly equal to model size * batch_size for deep networks as a very loose upper bound approximation per layer? 
        # No, that's inaccurate.
        # Better to just report static memory (weights+grads+optim) and warn about activations.
        
        static_mem = model_mem + grad_mem + optim_mem
        
        if mixed_precision:
            # Mixed precision might keep a copy of master weights in fp32 + fp16 weights/grads
            # This is a simplified view.
            pass

        return {
            "parameters_mb": model_mem / (1024**2),
            "gradients_mb": grad_mem / (1024**2),
            "optimizer_mb": optim_mem / (1024**2),
            "total_static_mb": static_mem / (1024**2),
            "note": "This estimate does not include activation memory, which scales with batch size and network depth."
        }

    def estimate_compute(self, input_sample):
        """
        Estimates FLOPs using hooks for a single forward pass.
        Supported layers: Linear, Conv2d, MultiheadAttention (approx).
        """
        flops = 0
        
        def count_flops(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                # input: (N, *, in_features)
                # output: (N, *, out_features)
                # FLOPs = 2 * in_features * out_features * batch_size * ...
                # We calculate per sample, so ignore batch dimension if possible or normalize later.
                # Let's calculate total for the input batch.
                inp = input[0]
                # macs = in * out * prod(other_dims)
                # flops = 2 * macs
                batch_size = inp.shape[0]
                other_dims = np.prod(inp.shape[1:-1])
                in_features = module.in_features
                out_features = module.out_features
                flops += 2 * batch_size * other_dims * in_features * out_features
                
            elif isinstance(module, nn.Conv2d):
                # input: (N, Cin, Hin, Win)
                # output: (N, Cout, Hout, Wout)
                # kernel: (Cout, Cin, K, K)
                # FLOPs = 2 * Cout * Cin * K * K * Hout * Wout * N
                inp = input[0]
                out = output
                batch_size = inp.shape[0]
                output_dims = np.prod(out.shape[2:])
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                flops += 2 * batch_size * module.out_channels * output_dims * kernel_ops
                
            elif isinstance(module, nn.MultiheadAttention):
                # Simplified estimation for MHA
                # Q, K, V projections + Attention + Output projection
                # input is usually (L, N, E) or (N, L, E)
                q, k, v = input[0], input[1], input[2]
                # Assuming q=k=v for self attention if only 1 input, but hook gets all args.
                # If packed in one arg, it's harder.
                # Let's assume standard usage.
                
                # This is hard to hook perfectly without knowing exact inputs.
                # We will skip complex modules and rely on sub-modules (Linear) if they are registered.
                # nn.MultiheadAttention uses internal Linear layers for q,k,v,out. 
                # If we hook recursively, we catch those Linear layers!
                # So we don't need to handle MultiheadAttention explicitly if we hook all Linears.
                pass

        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(count_flops))

        # Run forward pass
        with torch.no_grad():
            self.model(*input_sample)

        # Remove hooks
        for h in hooks:
            h.remove()

        return flops

    def print_summary(self, batch_size=1, input_sample=None):
        print("="*30)
        print("Model Size Estimator Summary")
        print("="*30)
        
        params = self.get_parameter_count()
        print(f"Total Parameters: {params:,}")
        print(f"Model Size (float32): {self.get_parameter_size_mb():.2f} MB")
        
        mem_est = self.estimate_memory_requirements(batch_size)
        print(f"\nStatic Memory Estimation (Batch Size {batch_size}):")
        print(f"  Model Weights: {mem_est['parameters_mb']:.2f} MB")
        print(f"  Gradients: {mem_est['gradients_mb']:.2f} MB")
        print(f"  Optimizer (Adam): {mem_est['optimizer_mb']:.2f} MB")
        print(f"  Total Static: {mem_est['total_static_mb']:.2f} MB")
        print(f"  (Note: {mem_est['note']})")
        
        if input_sample is not None:
            flops = self.estimate_compute(input_sample)
            print(f"\nComputational Complexity (Forward Pass):")
            print(f"  Total FLOPs: {flops:,}")
            print(f"  GFLOPs: {flops / 1e9:.4f}")
        
        print("="*30)


def test_estimation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLA_transformer().to(device)
    
    estimator = ModelSizeEstimator(model)
    
    # Create dummy input
    batch_size = 1
    img1 = torch.randn(batch_size, 3, 224, 224).to(device)
    img2 = torch.randn(batch_size, 3, 224, 224).to(device)
    img3 = torch.randn(batch_size, 3, 224, 224).to(device)
    state = torch.randn(batch_size, 3).to(device)
    
    input_sample = (img1, img2, img3, state)
    
    estimator.print_summary(batch_size=64, input_sample=input_sample)

if __name__ == "__main__":
    test_estimation()
