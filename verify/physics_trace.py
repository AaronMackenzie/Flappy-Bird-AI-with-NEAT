"""Print deterministic physics traces from headless_train.py as JSON.

verify/check_engine.js replays the same policies and pipe heights through the
JavaScript engine used by docs/index.html and compares the results.

    python verify/physics_trace.py > trace.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headless_train import play_policy  # noqa: E402

HEIGHTS = {
    "mid": [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
    "mixed": [300, 120, 400, 250, 60, 440, 180, 350, 90, 420, 200, 310, 140, 380, 260, 70],
    "hard": [50, 449, 50, 449, 50, 449, 50, 449],
}


def policy_never(y, top, bottom):
    return False


def policy_low(y, top, bottom):
    return y > 400


def make_every(n):
    state = {"i": 0}

    def p(y, top, bottom):
        state["i"] += 1
        return state["i"] % n == 0
    return p


def policy_gap(y, top, bottom):
    """Cheating-free but simple: hover just above the lower pipe's edge."""
    return y + 48 > bottom - 20


POLICIES = {"never": lambda: policy_never, "low": lambda: policy_low,
            "every17": lambda: make_every(17), "every9": lambda: make_every(9),
            "gap": lambda: policy_gap}

if __name__ == "__main__":
    out = []
    for pname, mk in POLICIES.items():
        for hname, hs in HEIGHTS.items():
            r = play_policy(mk(), hs, max_frames=3000)
            out.append({"policy": pname, "heights": hname, **r})
    print(json.dumps(out, indent=1))
