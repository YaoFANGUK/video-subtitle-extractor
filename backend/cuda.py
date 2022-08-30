import torch

cudaIsAvailable = torch.cuda.is_available()

print("cuda" if torch.cuda.is_available() else "cpu")

exit()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# conda create -n torch python=3.8

# mps_device = torch.device("cpu")
mps_device = torch.device("mps")

# Create a Tensor directly on the mps device
x = torch.ones(5, device=mps_device)
# Or
x = torch.ones(5, device="mps")

# Any operation happens on the GPU
y = x * 2

# # Move your model to mps just like any other device
# model = YourFavoriteNet()
# model.to(mps_device)

# # Now every call runs on the GPU
# pred = model(x)

x = torch.rand((10000,10000), dtype=torch.float32)
y = torch.rand((10000,16000), dtype=torch.float32)
x = x.to(mps_device)
y = y.to(mps_device)

print(list(mps_device))


# Send you tensor to GPU
# my_tensor = my_tensor.to(mps_device)