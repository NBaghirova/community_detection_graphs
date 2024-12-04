import numpy as np
from gurobipy import Model, GRB, quicksum

def find_connected_max_community(A):
    """
    Finds the connected maximum community (PDS) in a graph.

    Parameters:
    - A: numpy.ndarray
        Adjacency matrix of the graph.

    Returns:
    - dict: A dictionary containing:
        - "community": List of vertices in the connected max community.
        - "size": Size of the community.
    - str: If no solution is found, returns an appropriate message.
    """
    
    # Input validation
    assert A.shape[0] == A.shape[1], "Adjacency matrix must be square."
    assert (A == A.T).all(), "Adjacency matrix must be symmetric."

    n = A.shape[0]  # Number of vertices
    if n < 3:
        return "Graph has less than 3 vertices."

    V = range(n)

    # Precompute degrees
    d = np.sum(A, axis=1)

    # Create the Gurobi model
    model = Model("Connected_Max_Community")
    model.setParam("OutputFlag", 1)

    # Variables
    x = model.addVars(V, vtype=GRB.BINARY, name="x")  # 1 if vertex is in S
    f = model.addVars(V, V, vtype=GRB.CONTINUOUS, lb=0, name="f")  # Flow variables

    # Objective: Maximize the size of S
    model.setObjective(quicksum(x[i] for i in V), GRB.MAXIMIZE)

    # Constraints
    model.addConstr(quicksum(x[i] for i in V) >= 2, name="min_size_S")  # S has at least 2 vertices
    model.addConstr(quicksum(x[i] for i in V) <= n - 1, name="max_size_S")  # S excludes at least one vertex

    # Connectivity constraints
    root = 0
    model.addConstr(x[root] == 1, name="root_in_community")
    for i in V:
        if i != root:
            model.addConstr(
                quicksum(f[j, i] for j in V if A[j, i] == 1) ==
                quicksum(f[i, j] for j in V if A[i, j] == 1),
                name=f"flow_conservation_{i}"
            )
    for i in V:
        for j in V:
            if A[i, j] == 1:
                model.addConstr(f[i, j] <= x[i], name=f"flow_capacity_start_{i}_{j}")
                model.addConstr(f[i, j] <= x[j], name=f"flow_capacity_end_{i}_{j}")

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

    # Extract results
    if model.status == GRB.OPTIMAL:
        community = [i for i in V if x[i].X > 0.5]
        return {"community": community, "size": len(community)}
    else:
        return "Unexpected: No PDS found. Please check the model or graph input."
