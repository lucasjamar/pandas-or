from ortools.algorithms import pywrapknapsack_solver
import pandas as pd


def knapsack(items: pd.DataFrame, capacity: int) -> pd.Series:
    """
    Returns a boolean Series that determines whether an item of a given
    value and weight should be included in the knapsack.

        Parameters:
            items (pd.DataFrame): DataFrame of items containing a `value` and `weights` column.
            capacity (int): The capacity of the bin.

        Returns:
            is_included (pd.Series): Boolean Series determining whether an item should be included.

        Link:
            https://developers.google.com/optimization/bin/knapsack
    """
    if not {"value", "weight"}.issubset(items.columns):
        raise AttributeError("Items DataFrame must have 'value' and 'weight' columns.")
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "Knapsack",
    )
    solver.Init(items["value"], [items["weight"].tolist()], [capacity])
    solver.Solve()
    is_included = items.index.map(solver.BestSolutionContains)
    return is_included
