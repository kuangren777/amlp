"""S5 label handling shared by analysis/s5_label_eval.py and its test."""


def keep_labels(labels, excluded, ambiguous):
    """Drop labels on rows the plan-A rerun replaced (project lead). Ambiguous items keep their place in the
    labeller's numbering, so the 0-based indices are mapped onto the kept list."""
    keep = [i for i, x in enumerate(labels) if x["id"] not in excluded]
    return [labels[i] for i in keep], {j for j, i in enumerate(keep) if i in ambiguous}
