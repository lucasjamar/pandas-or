from ortools.algorithms import pywrapknapsack_solver
import pandas as pd


def knapsack(items: pd.DataFrame, capacity: int) -> pd.Series:
    if not {"value", "weight"}.issubset(items.columns):
        raise AttributeError("Items DataFrame must have 'value' and 'weight' columns.")
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "Knapsack",
    )
    solver.Init(items["value"], [items["weight"].tolist()], [capacity])
    solver.Solve()
    return items.index.map(solver.BestSolutionContains)
