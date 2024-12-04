import numpy as np
from gurobipy import Model, GRB, quicksum
from collections import defaultdict, deque

def find_k_community(A, k):
    """
    Finds k-community structure in a graph.

    Parameters:
    - A: numpy.ndarray
        Adjacency matrix of the graph.
    - k: int
        Number of communities.

    Returns:
    - dict: A dictionary of communities if feasible.
    - str: A message if no k-community exists.
    """
    assert A.shape[0] == A.shape[1], "Adjacency matrix must be square."
    assert (A == A.T).all(), "Adjacency matrix must be symmetric."

    n = A.shape[0]  # Number of vertices

    if n < 4:
        return "The graph has less than 4 vertices."
    if n < 2 * k:
        return "The number of vertices must be at least 2*k."
    if k < 2:
        return "k must be at least 2."
    
    V = range(n)

    # Initialize Gurobi model
    model = Model("k_community")

    # Variables
    x = model.addVars(V, range(k), vtype=GRB.BINARY, name="x")  # Vertex-to-community assignment
    w = model.addVars(V, V, range(k), range(k), vtype=GRB.BINARY, name="w")  # Interaction variables

    # Constraints

    # 1. Proportional density constraint using indicator constraints
    for i in V:
        for p in range(k):
            for q in range(k):
                if p != q:
                    N_C = quicksum(A[i, j1] * quicksum(w[j1, j2, p, q] for j2 in V) for j1 in V)
                    N_Cprime = quicksum(A[i, l1] * quicksum(w[l1, l2, q, p] for l2 in V) for l1 in V)
                    correction = quicksum(x[l, q] * A[i, l] for l in V)
                    
                    # Use an indicator constraint for conditional enforcement
                    model.addGenConstrIndicator(
                        x[i, p], 1,  # Enforce this constraint only if x[i, p] = 1
                        N_C >= N_Cprime - correction,
                        name=f"CpwrtCq_{i}_{p}_{q}"
                    )

    # 2. Link w to x
    for j1 in V:
        for j2 in V:
            for p in range(k):
                for q in range(k):
                    model.addConstr(w[j1, j2, p, q] <= x[j1, p], name=f"WlinkedtoX1_{j1}_{j2}_{p}_{q}")
                    model.addConstr(w[j1, j2, p, q] <= x[j2, q], name=f"WlinkedtoX2_{j1}_{j2}_{p}_{q}")
                    model.addConstr(w[j1, j2, p, q] >= x[j1, p] + x[j2, q] - 1, name=f"WlinkedtoX3_{j1}_{j2}_{p}_{q}")

    # 3. At least two vertices in each community
    for p in range(k):
        model.addConstr(quicksum(x[i, p] for i in V) >= 2, name=f"atLeastTwoinCi_{p}")

    # 4. Every vertex belongs to exactly one community
    for i in V:
        model.addConstr(quicksum(x[i, p] for p in range(k)) == 1, name=f"everyvertexbelongsto1community_{i}")

    # Optimize the model
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        return f"No {k}-community exists."
    else:
        # Extract results
        communities = {p + 1: [i for i in V if x[i, p].X > 0.5] for p in range(k)}
        return communities
