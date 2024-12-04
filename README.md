# Community Detection in Graphs

Implementations of graph community detection problems using Integer Linear Programming (ILP) with the Gurobi Optimizer.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Community Detection Problems](#community-detection-problems)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Directory Structure](#directory-structure)
6. [Usage](#usage)
7. [Running Tests](#running-tests)
8. [References](#references)

---

## Introduction

Community detection is a fundamental challenge in network analysis, aiming to cluster "similar" vertices into cohesive groups. The distinction between various clustering methods lies in how they define this similarity. In 2013, M. Olsen conducted experiments and proposed that incorporating proportionality into the definition of a community is both intuitive and reflective of real-world network behavior. This was supported further in the literature, for instance, in 2024 by Baghirova et al., who argued that approaches lacking the concept of proportionality fail to capture essential properties of communities in a "natural" way.

In this repository, we provide implementations for solving the following community detection problems using Integer Linear Programming (ILP) and the Gurobi Optimizer:

---

## Community Detection Problems

1. **k-Community Problem**: The objective of the k-community problem is to partition the vertices of a graph into communities such that each community contains at least two vertices, and each vertex has proportionally more neighbors within its own community than with respect to any other community. A generalized version of this problem, known as the generalized k-community problem, removes the requirement that each community must contain at least two vertices. The code for solving the generalized version can be easily adapted from the code for the k-community problem, contained in this repository, by simply removing the size constraint on the communities.

2. **Connected k-Community**: This problem extends the requirements of the k-community problem by additionally requiring that each community must induce a connected subgraph. Similarly, a connected generalized k-community can be derived by removing the size constraint on each community while retaining the connectivity requirement.

3. **Maximum Community**: This problem focuses on maximization. The goal is to identify a subgraph containing at least two vertices and at most n−1 vertices (where n is the total number of vertices in the graph) such that every vertex within the subgraph has proportionally more neighbors inside the subgraph than outside it.

4. **Connected Maximum Community**: This problem has the same requirements as the maximum community problem, with the additional condition that the identified subgraph must induce a connected subgraph.

We formulated these problems as Integer Linear Programming (ILP) models and leveraged Gurobi's efficient ILP solver to solve each problem. Each implementation is accompanied by a test file to validate the program's performance across various types of graphs and to address common input-related issues.

---

## Requirements

Before running the programs, ensure the following:
- **Python 3.8 or higher**
- **Gurobi Optimizer** installed and a valid license
- Required Python packages:
  - `numpy`
  - `gurobipy`

---

## Installation

### Clone the Repository
```
git clone https://github.com/NBaghirova/community_detection_graphs.git
cd community_detection_graphs
```
### Install the required dependencies:
```
pip install -r requirements.txt
```
---
## Directory Structure
The project directory is organized as follows:
```
community_detection_graphs/
├── src/
│   ├── k_community.py          
│   ├── connected_k_community.py 
│   ├── max_community.py
│   ├── connected_max_community.py
├── tests/
│   ├── test_k_community.py  
│   ├── test_connected_k_community.py 
│   ├── test_max_community.py
│   ├── test_connected_max_community.py
├── README.md         
└── requirements.txt      
```

---
## Usage

### Importing the Functions
You can use the functions in your own Python scripts by importing them from the src directory. Below is an example of k-community:
```
from src.k_community import find_k_community
import numpy as np

 Define the adjacency matrix
A = np.array([
**Your adjacency matrix**
])
k = 2
result = find_k_community(A, k)
print(f"Resulting k-communities: {result}")
```
---
## Running Tests

To ensure the correctness of the implementation, you can run the tests:
```
pytest tests/
```
---
## References
1. M. Olsen. A general view on computing communities. Mathematical Social Sciences, 66(3):331–336, 2013.
2. N. Baghirova and A. Castillon, Proportionally dense subgraphs: parameterized hardness and efficiently solvable cases, The 19th International Conference and Workshops on Algorithms and Computation, 2024

