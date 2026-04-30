from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from gurobipy import GRB, Constr, Model, quicksum, tupledict


def add_memory_peak_constraints(
    model: Model,
    *,
    memory_budget: float,
    forward_coeffs: Sequence[float],
    backward_coeffs: Sequence[float],
    x_vars: tupledict | None = None,
    constraint_prefix: str = "mem_peak",
) -> Tuple[tupledict, List[Constr]]:
    """
    Add the 32 simplified linear constraints k[t] <= memory_budget along with binary x[i].

    Each constraint currently uses the placeholder
        k[t] = coeff[t] * x[i]
    so that x[i] directly controls whether the stage consumes memory. You can later replace
    the placeholder with a more accurate expression.

    Parameters
    ----------
    model:
        Gurobi model instance.
    memory_budget:
        Memory limit Mbudget.
    forward_coeffs:
        16 coefficients describing the forward stages.
    backward_coeffs:
        16 coefficients describing the backward stages.
    x_vars:
        Optional existing tupledict/list of binary decision variables. Created when None.
    constraint_prefix:
        Prefix for the generated constraint names.

    Returns
    -------
    tupledict, List[Constr]
        - Binary variables x[i].
        - List of the 32 added constraints.
    """
    num_modules = len(forward_coeffs)
    if len(backward_coeffs) != num_modules:
        raise ValueError("forward_coeffs and backward_coeffs must have the same length (16).")

    if x_vars is None:
        x_vars = model.addVars(num_modules, vtype=GRB.BINARY, name="x")
    elif len(x_vars) != num_modules:
        raise ValueError("Provided x_vars length does not match number of modules.")

    constraints: List[Constr] = []

    for idx in range(num_modules):
        k_forward = forward_coeffs[idx] * x_vars[idx]
        constr = model.addConstr(k_forward <= memory_budget, name=f"{constraint_prefix}_f_{idx + 1}")
        constraints.append(constr)

    for idx in range(num_modules):
        k_backward = backward_coeffs[idx] * x_vars[idx]
        constr = model.addConstr(k_backward <= memory_budget, name=f"{constraint_prefix}_b_{idx + 1}")
        constraints.append(constr)

    return x_vars, constraints


def solve_memory_budget(
    *,
    memory_budget: float,
    forward_coeffs: Sequence[float],
    backward_coeffs: Sequence[float],
    objective_weights: Optional[Iterable[float]] = None,
    time_limit: Optional[float] = None,
    model_name: str = "memory_budget",
    constraint_prefix: str = "mem_peak",
):
    """
    Convenience wrapper that builds and solves the simplified checkpoint MILP.

    With the placeholder formulation the objective defaults to minimising ∑x[i]; override
    objective_weights if you prefer another linear objective.
    """
    model = Model(model_name)
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)

    x_vars, _ = add_memory_peak_constraints(
        model,
        memory_budget=memory_budget,
        forward_coeffs=forward_coeffs,
        backward_coeffs=backward_coeffs,
        constraint_prefix=constraint_prefix,
    )

    if objective_weights is None:
        objective = quicksum(x_vars[i] for i in range(len(x_vars)))
    else:
        weights = list(objective_weights)
        if len(weights) != len(x_vars):
            raise ValueError("objective_weights must have the same length as x_vars.")
        objective = quicksum(weights[i] * x_vars[i] for i in range(len(x_vars)))

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        raise ValueError("Memory budget model is infeasible.")

    solution = [int(round(x_vars[i].X)) for i in range(len(x_vars))]
    return solution, model


def _generate_inputs(seed: int = 0, tight: bool = False):
    import random

    random.seed(seed)
    forward_coeffs = [random.randint(200, 350) * 1024 * 1024 for _ in range(16)]
    backward_coeffs = [
        max(int(forward_coeffs[idx] * random.uniform(0.8, 1.1)), 1) for idx in range(16)
    ]

    baseline_budget = max(forward_coeffs + backward_coeffs)
    memory_budget = baseline_budget * (2 if not tight else 1)

    return dict(
        memory_budget=memory_budget,
        forward_coeffs=forward_coeffs,
        backward_coeffs=backward_coeffs,
    )


def _run_example(description: str, *, tight: bool = False, objective_weights=None):
    params = _generate_inputs(seed=42 if not tight else 24, tight=tight)
    if objective_weights is not None:
        params["objective_weights"] = objective_weights
    solution, _ = solve_memory_budget(**params)
    kept = sum(solution)
    print(f"[{description}] keep {kept}/16 activations -> {solution}")


if __name__ == "__main__":
    _run_example("Balanced budget (minimise keeps)", tight=False)
    _run_example("Balanced budget (maximise keeps)", tight=False, objective_weights=[-1.0] * 16)
    _run_example("Tight budget", tight=True)

