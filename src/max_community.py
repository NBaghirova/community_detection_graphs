import numpy as np
from gurobipy import Model, GRB, quicksum

def find_max_community(A):
    """
    Finds the maximum proportionally dense subgraph (PDS) of a given graph.

    Parameters:
    - A: numpy.ndarray
        Adjacency matrix of the graph.

    Returns:
    - dict: A dictionary containing:
        - "PDS": List of vertices in the PDS.
        - "size": Size of the PDS.
    - str: If the graph has fewer than 3 vertices or if no PDS is found, returns an appropriate message.
    """
    # Input validation
    assert A.shape[0] == A.shape[1], "Adjacency matrix must be square."
    assert (A == A.T).all(), "Adjacency matrix must be symmetric."

    n = A.shape[0]  # Number of vertices

    if n < 3:
        return "Graph has less than 3 vertices."

    V = range(n)

    # Create the Gurobi model
    model = Model("MaxCommunity")
    model.setParam("Threads", 4) 

    # Variables: x[i] = 1 if vertex i is in S, 0 otherwise
    x = model.addVars(V, vtype=GRB.BINARY, name="x")

    # Objective: Maximize the size of S
    model.setObjective(quicksum(x[i] for i in V), GRB.MAXIMIZE)

    # Constraints on the size of S
    model.addConstr(quicksum(x[i] for i in V) >= 2, name="min_size_S")  # S must have at least 2 vertices
    model.addConstr(quicksum(x[i] for i in V) <= n - 1, name="max_size_S")  # S must exclude at least one vertex

    # Community constraints
    for u in V:
        d_S = quicksum(A[u, v] * x[v] for v in V if u != v)
        d_not_S = quicksum(A[u, v] * (1 - x[v]) for v in V if u != v)
        S_size = quicksum(x[v] for v in V)
        complement_size = n - S_size

        # Enforce the proportional density inequality only if x[u] = 1
        model.addGenConstrIndicator(
            x[u], 1,  # Activate this constraint when x[u] = 1
            d_S * complement_size >= d_not_S * (S_size - 1),
            name=f"pds_constraint_{u}"
        )


    # Solve the model
    model.optimize()

    # Check results
    if model.status == GRB.OPTIMAL:
        S = [i for i in V if x[i].X > 0.5]
        return {"PDS": S, "size": len(S)}
    else:
        return "Unexpected: No PDS found. Please check the model or graph input."
