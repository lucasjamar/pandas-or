from ortools.linear_solver import pywraplp
import pandas as pd


def bin_packing(items: pd.DataFrame, capacity: int) -> pd.DataFrame:
    if not {"weight", "itemId"}.issubset(items.columns):
        raise AttributeError("Items DataFrame must have 'weight' and 'itemId' columns.")
    items = items.copy()
    bins = pd.DataFrame({"capacity": [capacity]}, index=items.index)
    bins["binId"] = bins.index
    items = items.merge(bins, how="cross")

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    items["itemIsPacked"] = items.apply(
        lambda x: solver.IntVar(0, 1, f'x_{x["itemId"]}_{x["binId"]}'), axis=1
    )
    # determine if bin is used
    bin_is_used = {x: solver.IntVar(0, 1, f"y[{x}]") for x in bins["binId"]}
    items["binIsUsed"] = items["binId"].map(bin_is_used)

    # Constraints. Each item must be in exactly one bin.
    for item_id, df_item in items.groupby("itemId"):
        solver.Add(df_item["itemIsPacked"].sum() == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for [bin_id, capacity], df_bin in items.groupby(["binId", "capacity"]):
        solver.Add(
            (df_bin["itemIsPacked"] * df_bin["weight"]).sum()
            <= bin_is_used[bin_id] * capacity
        )

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum(list(bin_is_used.values())))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        bin_is_used = {
            key: value.solution_value() for key, value in bin_is_used.items()
        }
        items["binIsUsed"] = items["binId"].map(bin_is_used) == 1
        items["itemIsPacked"] = (
            items["itemIsPacked"].apply(lambda x: x.solution_value()) == 1
        )
        items = items.query("binIsUsed and itemIsPacked").reset_index(drop=True)
        items["binId"] = items.groupby("binId").ngroup()
        return items.sort_values(["itemId", "binId"])
    else:
        print("The problem does not have an optimal solution.")
