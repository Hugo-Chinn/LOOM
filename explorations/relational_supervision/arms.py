"""Experiment arms shared by the portable and historical runners.

One arm changes one loss term. Combination arms are for phase 3 only.
"""

DEFAULT_WEIGHT = 0.1

ARMS = {
    "full": {},
    "sib_margin": {"sibling_margin_weight": DEFAULT_WEIGHT},
    "sib_decor": {"sibling_decor_weight": DEFAULT_WEIGHT},
    "edge_gain": {"edge_gain_weight": DEFAULT_WEIGHT},
    "edge_distill": {"edge_distill_weight": DEFAULT_WEIGHT},
    "combo_margin_gain": {"sibling_margin_weight": DEFAULT_WEIGHT,
                          "edge_gain_weight": DEFAULT_WEIGHT},
    "combo_all": {"sibling_margin_weight": DEFAULT_WEIGHT,
                  "sibling_decor_weight": DEFAULT_WEIGHT,
                  "edge_gain_weight": DEFAULT_WEIGHT,
                  "edge_distill_weight": DEFAULT_WEIGHT},
}
PHASE1 = ("full", "sib_margin", "sib_decor", "edge_gain", "edge_distill")


def expand(names, scales=(1.0,), margin=None):
    """Return {arm_name: overrides}. Scales multiply every *_weight of an arm.

    `full` is never scaled. A non-unit scale appends `_x<scale>` to the name.
    """
    result = {}
    for name in names:
        if name not in ARMS:
            raise ValueError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
        for scale in ((1.0,) if name == "full" else scales):
            if scale <= 0:
                raise ValueError("scales must be positive")
            overrides = {k: round(v * scale, 10) for k, v in ARMS[name].items()}
            if margin is not None and name != "full":
                if "sibling_margin_weight" in overrides:
                    overrides["sibling_margin"] = margin
                if "edge_gain_weight" in overrides:
                    overrides["edge_gain_margin"] = margin
            label = name if scale == 1.0 else f"{name}_x{scale:g}"
            if margin is not None and name != "full":
                label += f"_m{margin:g}"
            if label in result:
                raise ValueError(f"duplicate arm {label}")
            result[label] = overrides
    return result
