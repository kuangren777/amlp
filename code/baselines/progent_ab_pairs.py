"""Progent A/B pair list: banking + slack, benign + TripWire 2-injection rotation (same rule as progent_fidelity.pairs()).
Run once with the harness python: writes progent_ab_pairs.json = [[suite, ut, it|null], ...]."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness


def pairs():
    out = []
    for s in ("banking", "slack"):
        S = harness.suite(s)
        uts = sorted(S.user_tasks, key=lambda x: int(x.split("_")[-1]))
        its = sorted(S.injection_tasks, key=lambda x: int(x.split("_")[-1]))
        for i, ut in enumerate(uts):
            out.append((s, ut, None))
            for it in dict.fromkeys([its[i % len(its)], its[(i + len(its) // 2) % len(its)]]):
                out.append((s, ut, it))
    return out


if __name__ == "__main__":
    p = pairs()
    json.dump(p, open(f"{D}/progent_ab_pairs.json", "w"))
    print(len(p), "pairs;", sum(1 for x in p if x[2] is None), "benign;", sum(1 for x in p if x[2]), "attack")
