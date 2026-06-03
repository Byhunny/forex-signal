import torch, os
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"CPU cores: {os.cpu_count()}")
