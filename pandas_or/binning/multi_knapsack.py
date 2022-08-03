from ortools.linear_solver import pywraplp
import pandas as pd


def multi_knapsack(items: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    if not {"value", "weight", "itemId"}.issubset(items.columns):
        raise AttributeError("Items DataFrame must have 'value', 'weight' and 'itemId' columns.")
    if "binId" not in bins.columns:
        raise AttributeError("Bins DataFrame must have 'binId' column.")
    items, bins = items.copy(), bins.copy()
    items = items.merge(bins, how="cross")
    del bins

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    items["isPacked"] = items.apply(
        lambda x: solver.BoolVar(f'x_{x["itemId"]}_{x["binId"]}'), axis=1
    )

    # Constraints. Each item is assigned to at most one bin.
    for item_id, item_df in items.groupby(items.index):
        solver.Add(item_df["isPacked"].sum() <= 1)

    # The amount packed in each bin cannot exceed its capacity.
    for [bin_capacity, _], bin_df in items.groupby(["capacity", "binId"]):
        solver.Add((bin_df["isPacked"] * bin_df["weight"]).sum() <= bin_capacity)

    # Objective. Maximize total value of packed items.
    objective = solver.Objective()
    items.apply(lambda x: objective.SetCoefficient(x["isPacked"], x["value"]), axis=1)
    objective.SetMaximization()
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        items["isPacked"] = (
            items["isPacked"].apply(lambda x: x.solution_value()).astype(bool)
        )
        return items
    else:
        print("The problem does not have an optimal solution.")
