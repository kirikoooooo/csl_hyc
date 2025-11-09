from __future__ import annotations

import random
from typing import List

from gurobipy import GRB

from solver import solve_memory_budget


def _generate_synthetic_inputs(seed: int = 0) -> dict:
    random.seed(seed)

    # 16 modules -> 16 retained activations / backward peaks, 17 forward peaks (include final stage)
    retained = [random.randint(40, 120) * 1024 * 1024 for _ in range(16)]
    forward_peaks = [random.randint(200, 350) * 1024 * 1024 for _ in range(17)]

    backward_peaks = []
    for idx in range(16):
        # backward peak roughly scales with forward peak of same module
        forward_idx = min(idx, len(forward_peaks) - 1)
        backward_scale = random.uniform(0.8, 1.2)
        backward_peaks.append(int(forward_peaks[forward_idx] * backward_scale))

    return {
        "memory_budget": max(forward_peaks) + sum(retained) // 3,
        "global_pre_forward_mem": forward_peaks[0] // 5,
        "forward_peak_values": forward_peaks,
        "backward_peak_values": backward_peaks,
        "retained_activation_values": retained,
    }


def _print_solution(solution: List[int], description: str):
    keep_count = sum(solution)
    print(f"{description}: keep {keep_count} / {len(solution)} activations")
    print("x =", solution)


def test_balanced_budget():
    """
    Memory budget large enough to keep a handful of activations.
    Expect some 1s in solution but not all.
    """
    params = _generate_synthetic_inputs(seed=42)
    solution, model = solve_memory_budget(**params)
    _print_solution(solution, "Balanced budget")
    assert 0 < sum(solution) < len(solution), "Expect a mix of kept and dropped activations."
    assert model.Status == GRB.OPTIMAL


def test_tight_budget():
    """
    Very small budget should force all x[i] = 0 (no activations stored).
    """
    params = _generate_synthetic_inputs(seed=24)
    params["memory_budget"] = params["global_pre_forward_mem"] + max(params["forward_peak_values"]) // 2
    solution, model = solve_memory_budget(**params)
    _print_solution(solution, "Tight budget")
    assert sum(solution) == 0, "Tight budget should drop all activations."
    assert model.Status == GRB.OPTIMAL


if __name__ == "__main__":
    test_balanced_budget()
    test_tight_budget()
