"""FuncCode-KAN: codebook-based functional compression and hardware-aware
quantization for Kolmogorov-Arnold Networks.

Subpackages
-----------
- funcodekan.data          MNIST / CIFAR data loaders
- funcodekan.models        Dense and compressed KAN models (SplineKAN, FastKAN,
                           GRAM/KAGN, MLP baseline, SRB / index-efficient /
                           soft-to-hard variants)
- funcodekan.compression   Coefficient-, function-space, and branch-aware
                           codebook clustering
- funcodekan.quantization  Hardware-aware codebook quantization pipeline
- funcodekan.analysis      Bit-exact storage accounting
- funcodekan.hwq           Standalone HWQ library (bit-packing, export format)
- funcodekan.experiments   Runnable experiment drivers (python -m ...)
"""

__version__ = "1.0.0"
