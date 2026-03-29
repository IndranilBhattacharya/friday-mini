import torch

def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """
    Prints debugging information for a given tensor.
    
    In:
        name: str - identifier for the tensor (shape: none/scalar)
        tensor: torch.Tensor - tensor to inspect (shape: any)
        
    Out:
        None (shape: none/scalar)
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"[{name}] Not a tensor. Type: {type(tensor)}")
        return
        
    # .item() requires a scalar tensor, but we can call it on min() and max()
    # Handle potentially empty tensors gracefully
    if tensor.numel() == 0:
        print(f"[{name}] Shape: {list(tensor.shape)} | Dtype: {tensor.dtype} | Min: N/A (Empty) | Max: N/A (Empty)")
        return
        
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    
    print(f"[{name}] Shape: {list(tensor.shape)} | Dtype: {tensor.dtype} | Min: {t_min:.4f} | Max: {t_max:.4f}")
