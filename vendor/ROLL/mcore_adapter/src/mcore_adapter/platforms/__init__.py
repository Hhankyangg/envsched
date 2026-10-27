import torch
from .platform import Platform
from .cuda import CudaPlatform
from .cpu import CpuPlatform

current_platform: Platform = CudaPlatform() if torch.cuda.is_available() else CpuPlatform()
__all__ = ["Platform", "current_platform"]
