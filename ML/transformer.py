import torch
from torch_geometric.data import Data
import numpy as np
import os

def pad_tensor(tensor, target_size, dim=0):
    """
    pad the tensor to the desired size.
    """
    pad_size = list(tensor.size())
    pad_size[dim] = target_size - pad_size[dim]
    padding = torch.zeros(*pad_size, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def lp_to_bipartite_graph(c, A, b, max_vars, max_constraints):
    """
    Convert LP to bipartite graph
    """
    num_vars = 1200
    num_constraints = len(b)
    
    # Calculate normalization parameters
    c_min, c_max = min(c), max(c)
    b_min, b_max = min(b), max(b)
    
    # Variable node features
    variable_features = []
    for i in range(num_vars):
        feature1_var = normalize(c[i], c_min, c_max)
        nnz_ratio_var = np.count_nonzero(A[:, i])/ num_constraints

        variable_features.append([feature1_var, nnz_ratio_var])

    variable_features = torch.tensor(variable_features, dtype=torch.float)
    
    # Constraint node features
    constraint_features = []
    for j in range(num_constraints):
        constr_rhs = normalize(b[j], b_min, b_max)
        nnz_ratio_constraint = np.count_nonzero(A[j])/ num_vars

        constraint_features.append([constr_rhs, nnz_ratio_constraint])
    
    constraint_features = torch.tensor(constraint_features, dtype=torch.float)
    
    # Padding node features
    variable_features = pad_tensor(variable_features, max_vars, dim=0)
    constraint_features = pad_tensor(constraint_features, max_constraints, dim=0)
    
    # Edges
    edge_index = []
    edge_attr = []
    
    for i in range(num_constraints):
        for j in range(num_vars):
            edge_index.append([j, num_vars + i])
            edge_attr.append(A[i][j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    labels = 0
    
    data = Data(x=torch.cat([variable_features, constraint_features], dim=0),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=labels)
    
    return data