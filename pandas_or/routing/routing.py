from typing import Union
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def single_vehicle_route(
    routing,
    manager,
    solution,
    vehicle_id: int,
    depots: pd.DataFrame = None,
    with_load: bool = False,
) -> pd.DataFrame:
    departure_points, distances = [], []
    index = routing.Start(vehicle_id)
    loads = []
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        departure_points.append(manager.IndexToNode(previous_index))
        distances.append(
            routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        )
        if with_load:
            loads.append(depots.loc[previous_index, "demand"])
    departure_points.append(manager.IndexToNode(previous_index))
    distances.append(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))
    vehicle_route = pd.DataFrame(
        {
            "departure": departure_points,
            "distance": distances,
        }
    )
    if with_load:
        loads.append(depots.loc[previous_index, "demand"])
        vehicle_route["load"] = loads
    return vehicle_route


def single_vehicle_features(route: pd.DataFrame) -> pd.DataFrame:
    route["destination"] = route["departure"].shift(-1)
    route["tripsSinceStart"] = route.reset_index(drop=True).index
    route["tripsTillEnd"] = (
            route["tripsSinceStart"].max()
            - route["tripsSinceStart"]
    )
    route["distanceSinceStart"] = (
            route["distance"].cumsum() - route["distance"]
    )
    route["distanceTillEnd"] = (
            route["distanceSinceStart"].max()
            - route["distanceSinceStart"]
    )
    return route


def multi_vehicle_features(route: pd.DataFrame) -> pd.DataFrame:
    route["destination"] = route.groupby("vehicleId")["departure"].shift(-1)
    route["tripsSinceStart"] = route.groupby("vehicleId")["departure"].cumcount()
    route["tripsTillEnd"] = (
            route.groupby("vehicleId")["tripsSinceStart"].transform(max)
            - route["tripsSinceStart"]
    )
    route["distanceSinceStart"] = (
            route.groupby("vehicleId")["distance"].cumsum() - route["distance"]
    )
    route["distanceTillEnd"] = (
            route.groupby("vehicleId")["distanceSinceStart"].transform(max)
            - route["distanceSinceStart"]
    )
    return route


def solve_routing(
    df: pd.DataFrame,
    vehicles: Union[int, pd.DataFrame] = 1,
    locations: pd.DataFrame = None,
    starting_point: Union[str, int] = 0,
) -> pd.DataFrame:
    column_names = df.columns
    if isinstance(vehicles, pd.DataFrame):
        if "vehicleId" not in vehicles.columns:
            raise AttributeError("Items DataFrame must have 'vehicleId' column.")
        vehicle_columns = vehicles.columns
        num_vehicles = len(vehicles.index)
        vehicle_ids = vehicles["vehicleId"].tolist()
        if "capacity" in vehicle_columns:
            vehicle_capacities = vehicles["capacity"].tolist()
    else:
        num_vehicles = vehicles
        vehicle_ids = range(num_vehicles)
        vehicle_columns, vehicle_capacities = [], None
    if isinstance(starting_point, str):
        starting_point = column_names.get_loc(starting_point)
    manager = pywrapcp.RoutingIndexManager(
        len(column_names), num_vehicles, starting_point
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return df.iloc[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return locations.loc[from_node, "demand"]

    if num_vehicles > 1:
        if "capacity" in vehicle_columns:
            demand_callback_index = routing.RegisterUnaryTransitCallback(
                demand_callback
            )
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                vehicle_capacities,  # vehicle maximum capacities
                True,  # start cumul to zero
                "Capacity",
            )
        else:
            # Add Distance constraint.
            dimension_name = "Distance"
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                3000,  # vehicle maximum travel distance
                True,  # start cumul to zero
                dimension_name,
            )
            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Only for capacity
    if "capacity" in column_names:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        column_mapping = {index: value for index, value in enumerate(column_names)}
        if num_vehicles <= 1:
            route = single_vehicle_route(routing, manager, solution, 0)
        else:
            route = []
            for vehicle_id in vehicle_ids:
                if locations is not None:
                    vehicle_route = single_vehicle_route(
                        routing, manager, solution, vehicle_id, locations
                    )
                else:
                    vehicle_route = single_vehicle_route(
                        routing, manager, solution, vehicle_id
                    )
                vehicle_route["vehicleId"] = vehicle_id
                route.append(vehicle_route)
            route = pd.concat(route, ignore_index=True)
        route["departure"] = route["departure"].map(column_mapping)
        if num_vehicles <= 1:
            route = single_vehicle_features(route)
        else:
            route = multi_vehicle_features(route)
        return route
    else:
        raise ValueError("No solution found")
