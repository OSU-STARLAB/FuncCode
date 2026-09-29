import torch
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CompressedKANState:
    """
    Container for clustered/compressed KAN state.
    """
    codebooks: Dict[int, torch.Tensor]
    cluster_ids: Dict[int, torch.Tensor]
    other_state: Dict[str, torch.Tensor]

    @staticmethod
    def from_state_dict(state: Dict[str, torch.Tensor]) -> "CompressedKANState":
        codebooks = {}
        cluster_ids = {}
        other = {}

        for k, v in state.items():
            if k.startswith("clustered_weights."):
                idx = int(k.split(".")[-1])
                codebooks[idx] = v
            elif k.startswith("cluster_ids."):
                idx = int(k.split(".")[-1])
                cluster_ids[idx] = v.to(torch.long).reshape(-1)
            else:
                other[k] = v

        if len(codebooks) == 0:
            raise ValueError("No clustered_weights.* tensors found.")
        if set(codebooks.keys()) != set(cluster_ids.keys()):
            raise ValueError("Mismatched clustered_weights.* and cluster_ids.* layer indices.")

        return CompressedKANState(codebooks=codebooks, cluster_ids=cluster_ids, other_state=other)

    def layer_indices(self) -> List[int]:
        return sorted(self.codebooks.keys())
