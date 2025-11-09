from __future__ import annotations

from typing import Any, Callable, List, Sequence, Tuple

from gurobipy import GRB, Constr, Model, quicksum, tupledict


ForwardExprBuilder = Callable[[int, Sequence], Any]
BackwardExprBuilder = Callable[[int, int, Sequence], Any]


def _default_forward_expr(_: int, x_vars: Sequence) -> Any:
    """Placeholder forward-stage k[t] expression."""
    return quicksum(x_vars)


def _default_backward_expr(_: int, __: int, x_vars: Sequence) -> Any:
    """Placeholder backward-stage k[t] expression."""
    return quicksum(x_vars)


def add_memory_peak_constraints(
    model: Model,
    memory_budget: float,
    *,
    forward_expr_builder: ForwardExprBuilder | None = None,
    backward_expr_builder: BackwardExprBuilder | None = None,
    forward_stages: int = 17,
    backward_stages: int = 16,
    stage_offset: int = 1,
    constraint_prefix: str = "mem_peak",
) -> Tuple[tupledict, List[Constr]]:
    """
    Create binary checkpoint decisions x[i] and add 33 linear memory constraints.

    Parameters
    ----------
    model:
        Gurobi model instance that already contains other problem variables.
    memory_budget:
        The constant right-hand side Mbudget used in every k[t] <= Mbudget constraint.
    forward_expr_builder:
        Callable that returns the linear expression for k[t] when t is in the forward phase.
        Signature: f(stage_id:int, x_vars:Sequence[gurobipy.Var]) -> linear expression.
        If omitted, a dummy linear form is used (simply sum(x)).
    backward_expr_builder:
        Callable that returns the linear expression for k[t] when t is in the backward phase.
        Signature: f(stage_id:int, module_idx:int, x_vars:Sequence[gurobipy.Var]) -> expression,
        where module_idx counts backward modules in reverse order (15 -> 0). If omitted a dummy
        form sum(x) is used.
    forward_stages:
        Number of forward stages, default 17 (t = 1 ... 17).
    backward_stages:
        Number of backward stages, default 16 (t = 18 ... 33).
    stage_offset:
        Optional offset for naming, default 1 so the first constraint is mem_peak_stage_1.
    constraint_prefix:
        Name prefix for the generated constraints.

    Returns
    -------
    tupledict, List[Constr]
        - The tupledict of binary decision variables x[i].
        - A list with all generated gurobipy constraints.
    """
    forward_expr_builder = forward_expr_builder or _default_forward_expr
    backward_expr_builder = backward_expr_builder or _default_backward_expr

    x_vars = model.addVars(backward_stages, vtype=GRB.BINARY, name="x")
    ordered_x: List = [x_vars[i] for i in range(backward_stages)]

    constraints: List[Constr] = []
    stage_id = stage_offset

    for local_stage in range(forward_stages):
        k_expr = forward_expr_builder(stage_id, ordered_x)
        constr = model.addConstr(k_expr <= memory_budget, name=f"{constraint_prefix}_stage_{stage_id}")
        constraints.append(constr)
        stage_id += 1

    for local_stage in range(backward_stages):
        module_idx = backward_stages - 1 - local_stage
        k_expr = backward_expr_builder(stage_id, module_idx, ordered_x)
        constr = model.addConstr(k_expr <= memory_budget, name=f"{constraint_prefix}_stage_{stage_id}")
        constraints.append(constr)
        stage_id += 1

    return x_vars, constraints

