"""Device selection helpers.

Picks the best available accelerator so the same code runs on a CUDA GPU,
an Apple Silicon Mac (MPS), or plain CPU without any changes.
"""
import torch


def get_device():
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
