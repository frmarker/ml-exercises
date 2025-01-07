import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders (MPS)
    print("MPS backend is available. Running on:", device)
else:
    print("MPS backend is not available. Using CPU.")

    