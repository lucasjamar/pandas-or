# Knapsack data
def bins():
    return _get_dataset("binning/bins")


def items_knapsack():
    return _get_dataset("binning/items_knapsack")


def items_multi():
    return _get_dataset("binning/items_multi")


def items_bin_packing():
    return _get_dataset("binning/items_bin_packing")


# Routing data
def locations():
    return _get_dataset("routing/locations")


def vehicles(with_capacity: bool = True):
    data = _get_dataset("routing/vehicles")
    if not with_capacity:
        data = data.drop(columns=["capacity"])
    return data


def _get_dataset(d):
    import pandas
    import os

    return pandas.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", f"{d}.csv",
        )
    )
