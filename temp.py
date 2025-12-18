# from safetensors.torch import load_file

# try:
#     state_dict = load_file("resnet_18_pretrained.safetensors")
#     print("Keys in safetensors file:")
#     for key in state_dict.keys():
#         print(key)
# except Exception as e:
#     print(f"Error loading safetensors file: {e}")
# from safetensors.torch import load_file

# try:
#     state_dict = load_file("resnet_18_pretrained.safetensors")
#     print("Keys in safetensors file:")
#     for key in state_dict.keys():
#         print(key)
# except Exception as e:
#     print(f"Error loading safetensors file: {e}")


import torch


model_dict = torch.load("checkpoint.pth")
print(f'model dict: {model_dict.keys()}')