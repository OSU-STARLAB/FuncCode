"""Hardware-acceleration extension package (additive; see docs/HW_DESIGN_CONTRACT.md).

Nothing in here modifies the paper-verified pipeline; every module imports
from it. Frozen numeric contract lives in hw_spec.py and must match
hw/hls/common/hw_config.h bit for bit.
"""
