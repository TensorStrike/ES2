import torch
from collections import OrderedDict


def print_state_dict_info(state_dict, max_params=10):
    print("\nState Dict Parameters:")
    count = 0
    total_params = 0
    total_nonzero = 0

    for name, tensor in state_dict.items():
        if count >= max_params:
            print("...")
            break
        nonzero = (tensor != 0).sum().item()
        total_params += tensor.numel()
        total_nonzero += nonzero
        print(f"{name}: shape {tensor.shape}, non-zero: {nonzero}/{tensor.numel()}")
        count += 1

    print(f"\nTotal sparsity: {1 - total_nonzero / total_params:.2%}")


# Load and analyze state dict
state_dict = torch.load('rigl5.pt')
print_state_dict_info(state_dict)