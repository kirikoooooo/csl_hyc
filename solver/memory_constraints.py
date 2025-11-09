from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple, Union

from gurobipy import Constr, Model, Var, quicksum


Number = Union[int, float]


def _as_ordered_var_list(x_vars: Union[Sequence[Var], Mapping, Iterable[Tuple[int, Var]]]) -> List[Var]:
    """Convert different gurobipy container types to an ordered list of variables."""
    if isinstance(x_vars, Mapping):
        items = list(x_vars.items())
        try:
            items.sort(key=lambda item: item[0])
        except TypeError:
            pass
        return [var for _, var in items]
    if hasattr(x_vars, "items"):
        items = list(x_vars.items())  # type: ignore[attr-defined]
        try:
            items.sort(key=lambda item: item[0])
        except TypeError:
            pass
        return [var for _, var in items]
    try:
        return list(x_vars)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("Unsupported container type for decision variables 'x_vars'.") from exc


def add_memory_peak_constraints(
    model: Model,
    x_vars: Union[Sequence[Var], Mapping],
    forward_peak_mem: Sequence[Number],
    backward_peak_mem: Sequence[Number],
    retained_activation: Sequence[Number],
    base_forward_mem: Number,
    memory_budget: Number,
    *,
    stage_offset: int = 1,
    constraint_prefix: str = "mem_peak",
) -> List[Constr]:
    """
    Add the 33 memory budget constraints k[t] <= Mbudget for the MILP.

    The function assumes:
    - x_vars contains 16 decision variables x[i] (binary or continuous) indicating whether
      the activation of module i is retained.
    - forward_peak_mem provides the transient peak memory of each forward stage. Pass a
      sequence of length 16 (one per module) or 17 if an additional post-forward stage
      should be modelled. When only 16 values are supplied, an extra zero-valued stage is
      appended automatically so that 33 constraints are generated.
    - backward_peak_mem provides the estimated peak memory during the backward pass per
      module, ordered the same way as x_vars.
    - retained_activation holds the activation size that is kept if x[i] = 1 for each module.

    Parameters
    ----------
    model:
        The gurobipy model to which the constraints are added.
    x_vars:
        Decision variables x[i] (length 16). Can be a list/tuple or a gurobipy tupledict.
    forward_peak_mem:
        Peak memory of the forward stages. Length should be len(x_vars) or len(x_vars) + 1.
    backward_peak_mem:
        Peak memory of backward stages (length len(x_vars)).
    retained_activation:
        Activation sizes that remain stored when a module is checkpointed (length len(x_vars)).
    base_forward_mem:
        The baseline memory usage before executing a forward stage (global_pre_forward_mem).
    memory_budget:
        The available memory budget Mbudget.
    stage_offset:
        Optional offset for stage numbering in constraint names (default 1).
    constraint_prefix:
        Prefix used when naming the generated constraints.

    Returns
    -------
    List[Constr]
        The list of gurobipy constraints added to the model.
    """
    x_list = _as_ordered_var_list(x_vars)
    num_modules = len(x_list)

    if len(retained_activation) != num_modules:
        raise ValueError(
            f"retained_activation length ({len(retained_activation)}) "
            f"must match number of decision variables ({num_modules})."
        )
    if len(backward_peak_mem) != num_modules:
        raise ValueError(
            f"backward_peak_mem length ({len(backward_peak_mem)}) "
            f"must match number of decision variables ({num_modules})."
        )
    if len(forward_peak_mem) not in (num_modules, num_modules + 1):
        raise ValueError(
            "forward_peak_mem length must be either equal to the number of decision "
            "variables or exactly one greater to account for an additional forward stage."
        )

    if len(forward_peak_mem) == num_modules:
        forward_peaks = list(forward_peak_mem) + [0]
    else:
        forward_peaks = list(forward_peak_mem)
    backward_peaks = list(backward_peak_mem)
    retained = list(retained_activation)

    constraints: List[Constr] = []
    stage_idx = stage_offset
    cumulative_final = 0

    # Forward stages (t = 1 ... 17)
    for module_idx, forward_peak in enumerate(forward_peaks):
        if module_idx < num_modules:
            cumulative_final += x_list[module_idx] * retained[module_idx]
        k_expr = base_forward_mem + forward_peak + cumulative_final
        constr = model.addConstr(
            k_expr <= memory_budget,
            name=f"{constraint_prefix}_stage_{stage_idx}",
        )
        constraints.append(constr)
        stage_idx += 1

    total_final = quicksum(x_list[i] * retained[i] for i in range(num_modules))
    cumulative_release = 0

    # Backward stages (t = 18 ... 33)
    for module_idx in range(num_modules - 1, -1, -1):
        k_expr = backward_peaks[module_idx] + total_final - cumulative_release
        constr = model.addConstr(
            k_expr <= memory_budget,
            name=f"{constraint_prefix}_stage_{stage_idx}",
        )
        constraints.append(constr)
        stage_idx += 1
        cumulative_release += x_list[module_idx] * retained[module_idx]

    return constraints

