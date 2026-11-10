"""Pre-registered arms, seeds, grids and decision constants (PLAN.md protocol).

Everything a decision depends on is fixed here before any real-data run and is
copied into each config directory's protocol.json. Changing a value after
results exist starts a new exploratory phase; it does not amend this one.

Frozen 2026-11-10 (owner-confirmed, before any real-data run).
"""

DEV_SEEDS = (20262027, 20262028, 20262029, 20262030)
CONFIRM_SEEDS = (20262031, 20262032, 20262033, 20262034)
WEIGHT_GRID = (0.03, 0.10, 0.30)
MARGIN_GRID = (0.0, 0.2, 0.5)
DEFAULT_WEIGHT = 0.10
DEFAULT_MARGIN = 0.2
MIN_POSITIVE_SEEDS = 3          # of the four development seeds, per target task
LOGLOSS_TOLERANCE_REL = 0.005   # max mean paired relative validation logloss rise

# term -> (weight key, margin key or None, relation that defines target tasks)
TERMS = {
    "sib_margin": ("sibling_margin_weight", "sibling_margin", "siblings"),
    "sib_decor": ("sibling_decor_weight", None, "siblings"),
    "edge_gain": ("edge_gain_weight", "edge_gain_margin", "children"),
    "edge_distill": ("edge_distill_weight", None, "children"),
}
BASELINE = "full"


def has_margin(term):
    return TERMS[term][1] is not None


def overrides(term, weight=DEFAULT_WEIGHT, margin=None):
    """RelSupConfig overrides of one term at one grid point."""
    if term not in TERMS:
        raise ValueError(f"unknown term {term!r}; known: {sorted(TERMS)}")
    weight_key, margin_key, _ = TERMS[term]
    if weight <= 0:
        raise ValueError("weight must be positive")
    values = {weight_key: float(weight)}
    if margin_key is not None:
        values[margin_key] = float(DEFAULT_MARGIN if margin is None else margin)
    elif margin is not None:
        raise ValueError(f"{term} has no margin")
    return values


def arm_name(term, weight=DEFAULT_WEIGHT, margin=None):
    """The phase-1 point keeps the bare term name; other points are suffixed."""
    if has_margin(term):
        margin = DEFAULT_MARGIN if margin is None else margin
        if weight == DEFAULT_WEIGHT and margin == DEFAULT_MARGIN:
            return term
        return f"{term}_w{weight:g}_m{margin:g}"
    return term if weight == DEFAULT_WEIGHT else f"{term}_w{weight:g}"


def _arm(term, weight=DEFAULT_WEIGHT, margin=None):
    margin = (DEFAULT_MARGIN if margin is None else margin) if has_margin(term) else None
    return arm_name(term, weight, margin), {
        "terms": [term], "weight": {term: weight}, "margin": {term: margin},
        "overrides": overrides(term, weight, margin)}


def phase1():
    """One term per arm at the default point, plus the control."""
    arms = {BASELINE: {"terms": [], "weight": {}, "margin": {}, "overrides": {}}}
    arms.update(dict(_arm(term) for term in TERMS))
    return arms


def phase2(survivors):
    """Remaining grid points of each surviving term (the default point ran in phase 1)."""
    arms = {}
    for term in survivors:
        if term not in TERMS:
            raise ValueError(f"unknown term {term!r}")
        margins = MARGIN_GRID if has_margin(term) else (None,)
        for weight in WEIGHT_GRID:
            for margin in margins:
                name, spec = _arm(term, weight, margin)
                if name != term:
                    arms[name] = spec
    return arms


def frozen_arms(frozen, combine=False):
    """Arms from a frozen selection {term: {"weight", "margin"}}.

    combine=False: one arm per selected term. combine=True: a single arm with
    every selected term at its frozen setting (needs at least two terms).
    """
    if not frozen:
        raise ValueError("selection froze no term")
    singles = dict(_arm(term, s["weight"], s.get("margin")) for term, s in sorted(frozen.items()))
    if not combine:
        return singles
    if len(singles) < 2:
        raise ValueError("a combination needs at least two frozen terms")
    merged = {"terms": [], "weight": {}, "margin": {}, "overrides": {}}
    for spec in singles.values():
        for key in merged:
            (merged[key].extend if key == "terms" else merged[key].update)(spec[key])
    return {"combo_" + "+".join(sorted(merged["terms"])): merged}


def target_tasks(terms, rel, tasks):
    """Tasks an arm is predicted to improve, derived from the graph only."""
    targets = set()
    for term in terms:
        if TERMS[term][2] == "siblings":
            targets |= {node for a, b, _ in rel.siblings for node in (a, b)}
        else:
            targets |= {child for child, _ in rel.edges}
    return [tasks[i] for i in sorted(targets)]


def protocol_constants():
    return {"dev_seeds": list(DEV_SEEDS), "confirm_seeds": list(CONFIRM_SEEDS),
            "weight_grid": list(WEIGHT_GRID), "margin_grid": list(MARGIN_GRID),
            "default_weight": DEFAULT_WEIGHT, "default_margin": DEFAULT_MARGIN,
            "min_positive_seeds": MIN_POSITIVE_SEEDS,
            "logloss_tolerance_rel": LOGLOSS_TOLERANCE_REL, "baseline": BASELINE}
