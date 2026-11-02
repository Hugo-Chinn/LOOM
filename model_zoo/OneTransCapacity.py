"""Native OneTrans refinement capacity study, with an unchanged full backbone."""
import torch
from torch import nn

from .OneTrans import OneTrans, NSRefinementBlock


class OneTransCapacity(OneTrans):
    def __init__(self, feature_map, refiner_expansion_factor=4,
                 capacity_bypass_tasks=None, **kwargs):
        self.capacity_refiner_expansion = int(refiner_expansion_factor)
        self.capacity_bypass_labels = list(capacity_bypass_tasks or [])
        self.capacity_constructor = dict(kwargs)
        self.capacity_base = None
        super().__init__(feature_map, **kwargs)

    def compile(self, *args, **kwargs):
        if self.model_type != "base_refiner" or self.refiner_mode != "recurrent":
            raise ValueError("Capacity study requires native recurrent base_refiner")
        unknown = set(self.capacity_bypass_labels) - set(self.feature_map.labels)
        if unknown:
            raise ValueError(f"Unknown bypass labels: {unknown}")
        p = self.capacity_constructor
        self.ns_refiner = NSRefinementBlock(
            input_dim=self.token_dim, num_heads=p.get("num_heads", 1),
            num_layers=self.refiner_num_layers, num_ns_token=self.num_ns_token,
            num_stages=self.num_loops, dnn_activations=p.get("dnn_activations", "ReLU"),
            expansion_factor=self.capacity_refiner_expansion,
            peft_type=self.peft_type, peft_rank=self.peft_rank,
            peft_alpha=self.peft_alpha, peft_dropout=self.peft_dropout,
        )
        self.capacity_bypass_gates = nn.ParameterDict({
            label: nn.Parameter(torch.zeros(())) for label in self.capacity_bypass_labels
        })
        for index, label in enumerate(self.feature_map.labels):
            if label in self.capacity_bypass_labels:
                self.tower[index].register_forward_pre_hook(self._bypass_hook(label))
        super().compile(*args, **kwargs)

    def _bypass_hook(self, label):
        def hook(module, inputs):
            if self.capacity_base is None:
                raise RuntimeError("Bypass requires a base reference")
            return (inputs[0] + torch.tanh(self.capacity_bypass_gates[label])
                    * self.capacity_base.flatten(start_dim=1),)
        return hook

    def _run_base_refiner(self, *args, **kwargs):
        result = super()._run_base_refiner(*args, **kwargs)
        self.capacity_base = result[1]
        return result
