from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from gurobipy import GRB, Constr, LinExpr, Model, quicksum, tupledict


def add_memory_peak_constraints(
    model: Model,
    *,
    memory_budget: float,
    global_pre_forward_mem: float,
    forward_peak_values: Sequence[float],
    backward_peak_values: Sequence[float],
    retained_activation_values: Sequence[float],
    x_vars: tupledict | None = None,
    stage_offset: int = 1,
    constraint_prefix: str = "mem_peak",
) -> Tuple[tupledict, List[Constr]]:
    """
    Add the 33 linear constraints k[t] <= memory_budget with binary decisions x[i].

    Parameters
    ----------
    model:
        Gurobi model instance.
    memory_budget:
        Memory limit Mbudget.
    global_pre_forward_mem:
        Global baseline memory before each forward stage.
    forward_peak_values:
        Sequence giving get_peak_mem(t) for forward stages (expected length 17).
    backward_peak_values:
        Sequence giving get_backward_peak(i) for backward stages (expected length 16).
    retained_activation_values:
        Sequence giving the retained activation size for each module i (length 16).
    x_vars:
        Optional existing tupledict of binary decision variables x[i]; created if None.
    stage_offset:
        Starting index used when naming constraints.
    constraint_prefix:
        Prefix for the constraint names.

    Returns
    -------
    tupledict, List[Constr]
        - Binary decision variables x[i].
        - The list of generated constraints.
    """
    num_modules = len(retained_activation_values)
    if len(backward_peak_values) != num_modules:
        raise ValueError("backward_peak_values length must match retained_activation_values length.")

    if x_vars is None:
        x_vars = model.addVars(num_modules, vtype=GRB.BINARY, name="x")
    elif len(x_vars) != num_modules:
        raise ValueError("Provided x_vars length does not match retained_activation_values length.")

    forward_peaks = list(forward_peak_values)
    backward_peaks = list(backward_peak_values)
    forward_stage_count = len(forward_peaks)
    backward_stage_count = len(backward_peaks)
    if forward_stage_count + backward_stage_count == 32:
        forward_peaks.append(0.0)
        forward_stage_count += 1
    if forward_stage_count + backward_stage_count != 33:
        raise ValueError("Expected 33 stages in total (forward + backward).")

    ordered_x = [x_vars[i] for i in range(num_modules)]
    constraints: List[Constr] = []
    stage_id = stage_offset

    def get_peak_mem(stage: int) -> float:
        return forward_peaks[stage - 1]

    def get_backward_peak(module_idx: int) -> float:
        return backward_peaks[module_idx - 1]

    def get_final_mem(module_idx: int) -> LinExpr:
        return ordered_x[module_idx - 1] * retained_activation_values[module_idx - 1]

    def get_release(module_idx: int) -> LinExpr:
        return ordered_x[module_idx - 1] * retained_activation_values[module_idx - 1]

    cum_final_expr: LinExpr = LinExpr()
    for t in range(1, forward_stage_count + 1):
        k_expr = global_pre_forward_mem + get_peak_mem(t) + cum_final_expr
        constraints.append(
            model.addConstr(k_expr <= memory_budget, name=f"{constraint_prefix}_stage_{stage_id}")
        )
        stage_id += 1
        if t <= num_modules:
            cum_final_expr += get_final_mem(t)

    total_final_expr = quicksum(get_final_mem(i + 1) for i in range(num_modules))
    cum_release_expr: LinExpr = LinExpr()

    for local_stage in range(backward_stage_count):
        module_idx = num_modules - local_stage
        k_expr = get_backward_peak(module_idx) + total_final_expr - cum_release_expr
        constraints.append(
            model.addConstr(k_expr <= memory_budget, name=f"{constraint_prefix}_stage_{stage_id}")
        )
        stage_id += 1
        cum_release_expr += get_release(module_idx)

    return x_vars, constraints


def solve_memory_budget(
    *,
    memory_budget: float,
    global_pre_forward_mem: float,
    forward_peak_values: Sequence[float],
    backward_peak_values: Sequence[float],
    retained_activation_values: Sequence[float],
    objective_weights: Optional[Iterable[float]] = None,
    time_limit: Optional[float] = None,
    model_name: str = "memory_budget",
    constraint_prefix: str = "mem_peak",
):
    """
    Convenience wrapper that builds and solves the MILP for the checkpoint decisions x[i].

    Parameters
    ----------
    memory_budget:
        Memory limit Mbudget.
    global_pre_forward_mem:
        Global baseline memory before each forward stage.
    forward_peak_values / backward_peak_values / retained_activation_values:
        Same semantics as in add_memory_peak_constraints.
    objective_weights:
        Optional weights to penalise keeping activations. If omitted, minimise sum(x).
    time_limit:
        Optional solver time limit (seconds).
    model_name:
        Name of the gurobipy model.
    constraint_prefix:
        Prefix used for naming the 33 constraints.

    Returns
    -------
    Tuple[List[int], Model]
        - Binary solution vector x[i] (rounded to {0,1}).
        - The solved gurobipy model instance (for further inspection).
    """
    model = Model(model_name)
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)

    x_vars, _ = add_memory_peak_constraints(
        model,
        memory_budget=memory_budget,
        global_pre_forward_mem=global_pre_forward_mem,
        forward_peak_values=forward_peak_values,
        backward_peak_values=backward_peak_values,
        retained_activation_values=retained_activation_values,
        stage_offset=1,
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
    retained = [random.randint(40, 120) * 1024 * 1024 for _ in range(16)]
    forward_peaks = [random.randint(200, 350) * 1024 * 1024 for _ in range(17)]
    backward_peaks = [
        int(forward_peaks[min(idx, len(forward_peaks) - 1)] * random.uniform(0.8, 1.2))
        for idx in range(16)
    ]

    global_pre_forward_mem = forward_peaks[0] // 5
    memory_budget = max(forward_peaks) + sum(retained) // 3
    if tight:
        memory_budget = global_pre_forward_mem + max(forward_peaks) // 2

    return dict(
        memory_budget=memory_budget,
        global_pre_forward_mem=global_pre_forward_mem,
        forward_peak_values=forward_peaks,
        backward_peak_values=backward_peaks,
        retained_activation_values=retained,
    )


def _run_example(description: str, tight: bool = False):
    params = _generate_inputs(seed=42 if not tight else 24, tight=tight)
    solution, _ = solve_memory_budget(**params)
    kept = sum(solution)
    print(f"[{description}] keep {kept}/16 activations -> {solution}")


if __name__ == "__main__":
    _run_example("Balanced budget", tight=False)
    _run_example("Tight budget", tight=True)

