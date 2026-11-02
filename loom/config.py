"""Strict, serializable configuration for the portable Native-style core."""
from dataclasses import asdict, dataclass, field
import math


@dataclass
class LOOMConfig:
    tasks: list[str]
    token_dim: int = 64
    num_tokens: int = 4
    topology: str = "dag"
    graph: dict[str, list[str]] = field(default_factory=dict)
    expansion_factor: int = 2
    update_layers: int = 1
    update_type: str = "mixer"
    update_options: dict = field(default_factory=dict)
    num_preloops: int = 0
    behavior_conditioning: bool = True
    depth_conditioning: bool = False
    shared_update: bool = True
    use_loop_gate: bool = True
    loop_gate_init: float = 0.5
    supervision: str = "deep_progressive"
    auxiliary_loss_weight: float = 0.5
    parent_fusion: str = "mean"
    task_loss_weights: list[float] | None = None
    base_bypass_tasks: list[str] = field(default_factory=list)
    dropout: float = 0.0

    def __post_init__(self):
        if not isinstance(self.update_type, str) or not self.update_type:
            raise ValueError("update_type must be a nonempty registered name")
        if not isinstance(self.update_options, dict):
            raise ValueError("update_options must be a mapping")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be a nonempty ordered list of unique names")
        if any(not isinstance(t, str) or not t for t in self.tasks):
            raise ValueError("task names must be nonempty strings")
        for name in ("token_dim", "num_tokens", "expansion_factor", "update_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_preloops, int) or self.num_preloops < 0:
            raise ValueError("num_preloops must be a nonnegative integer")
        for name in ("behavior_conditioning", "depth_conditioning", "shared_update",
                     "use_loop_gate"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.topology not in {"dag", "linear", "branch"}:
            raise ValueError("topology must be dag, linear, or branch")
        if self.topology == "branch" and len(self.tasks) != 4:
            raise ValueError("branch requires ordered tasks [Click, Cart, Fav, Buy]")
        if self.topology != "dag" and self.graph:
            raise ValueError("graph is only accepted for topology=dag")
        if self.topology == "dag":
            if set(self.graph) != set(self.tasks):
                raise ValueError("graph must declare every task, including roots")
            for task, parents in self.graph.items():
                if not isinstance(parents, list) or len(set(parents)) != len(parents):
                    raise ValueError(f"graph[{task}] needs unique parents in a list")
                if task in parents or not set(parents).issubset(self.tasks):
                    raise ValueError(f"invalid parents for {task}")
        if self.parent_fusion not in {"mean", "softmax", "first"}:
            raise ValueError("parent_fusion must be mean, softmax, or first")
        if self.topology == "branch" and self.parent_fusion == "softmax":
            raise ValueError("legacy branch uses residual sigmoid fusion, not DAG softmax")
        if self.supervision not in {"progressive", "deep_progressive"}:
            raise ValueError("supervision must be progressive or deep_progressive")
        if not math.isfinite(self.auxiliary_loss_weight) or self.auxiliary_loss_weight < 0:
            raise ValueError("auxiliary_loss_weight must be finite and nonnegative")
        if not 0 < self.loop_gate_init < 1:
            raise ValueError("loop_gate_init must be strictly between 0 and 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.task_loss_weights is not None:
            if len(self.task_loss_weights) != len(self.tasks):
                raise ValueError("one loss weight per ordered task is required")
            if any(not math.isfinite(w) or w <= 0 for w in self.task_loss_weights):
                raise ValueError("task loss weights must be finite and positive")
        if (len(set(self.base_bypass_tasks)) != len(self.base_bypass_tasks)
                or not set(self.base_bypass_tasks).issubset(self.tasks)):
            raise ValueError("base_bypass_tasks must be unique declared tasks")

    def to_dict(self):
        return asdict(self)
