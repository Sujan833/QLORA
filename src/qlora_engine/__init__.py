"""
Universal QLoRA Fine-Tuner Engine Package.
"""

import os
import sys
import types
import importlib.machinery

# Force single GPU execution environment to prevent Trainer from wrapping models in DataParallel
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Automatic guard against broken torchvision C++ binary extensions in Kaggle/Colab
# Uses DummyMeta to dynamically respond to any attribute access (e.g. InterpolationMode.BOX)
try:
    import torchvision
except Exception:
    class DummyMeta(type):
        def __getattr__(cls, name):
            return name.lower()

    class DummyInterpolationMode(metaclass=DummyMeta):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"
        NEAREST_EXACT = "nearest-exact"

    class DummyImageReadMode(metaclass=DummyMeta):
        UNCHANGED = 0

    tv_mock = types.ModuleType("torchvision")
    tv_mock.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)

    tv_transforms_mock = types.ModuleType("torchvision.transforms")
    tv_transforms_mock.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    tv_transforms_mock.InterpolationMode = DummyInterpolationMode

    tv_io_mock = types.ModuleType("torchvision.io")
    tv_io_mock.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)
    tv_io_mock.ImageReadMode = DummyImageReadMode
    tv_io_mock.decode_image = lambda *args, **kwargs: None

    tv_mock.transforms = tv_transforms_mock
    tv_mock.io = tv_io_mock

    sys.modules["torchvision"] = tv_mock
    sys.modules["torchvision.transforms"] = tv_transforms_mock
    sys.modules["torchvision.io"] = tv_io_mock

__version__ = "0.1.0"
