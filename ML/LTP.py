import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from ML.pivot_arch import *
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os
from scipy.optimize import linprog
import numpy as np
import time

class LTP:

    def __init__(self, path):
        """
        Initilise the model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PivotGCN(input_dim=3, hidden_dim=128, output_dim=2).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        checkpoint_path = path
        if os.path.exists(path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.eval()

    def load_lp(self, data):
        """
        load the file
        """
        c = data['c']
        A = data['A']
        b = data['b']
        basis = data['basis']

        return A, b, c, basis
    
    def normalize(self, x, min_val, max_val):
        """
        Normalise the data
        """
        return (x - min_val) / (max_val - min_val)
    
    #Return the uniform detail (just the features and edge)
    def bipartite_graph_details(self, c, A, b):
        """
        Convert data into a bipartite graph.
        """
        num_vars = 1200
        num_constraints = len(b)

        # Calculate normalization parameters
        c_min, c_max = min(c), max(c)
        b_min, b_max = min(b), max(b)

        # Variables
        variable_features = []
        for i in range(num_vars):
            feature1_var = self.normalize(c[i], c_min, c_max)
            nnz_ratio_var = np.count_nonzero(A[:, i]) / num_constraints
            variable_features.append([feature1_var, nnz_ratio_var])
        
        variable_features = torch.tensor(variable_features, dtype=torch.float)
        
        # Constraints
        constraint_features = []
        for j in range(num_constraints):
            constr_rhs = self.normalize(b[j], b_min, b_max)
            nnz_ratio_constraint = np.count_nonzero(A[j]) / num_vars
            constraint_features.append([constr_rhs, nnz_ratio_constraint])
        
        constraint_features = torch.tensor(constraint_features, dtype=torch.float)

        # Edges
        edge_index = []
        edge_attr = []

        for i in range(num_constraints):
            for j in range(num_vars):
                edge_index.append([j, num_vars + i])
                edge_attr.append(A[i][j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return variable_features, constraint_features, edge_index, edge_attr
    
    def get_bipartite(self, variable_features, constraint_features, edge_index, edge_attr, cur_basis):
        """
        Collect bipartite graph detail.
        """
        num_vars = 1200
        num_constraints = 700
        
        variable_features = torch.cat((variable_features, torch.tensor(cur_basis[:num_vars], dtype=torch.float).view(-1, 1)), dim=1)
        constraint_features = torch.cat((constraint_features, torch.tensor(cur_basis[num_vars:], dtype=torch.float).view(-1, 1)), dim=1)

        data = Data(x=torch.cat([variable_features, constraint_features], dim=0),
                    edge_index=edge_index,
                    edge_attr=edge_attr)
        return data
    
    def get_matches(self, result, optimal_basis):
        matching_ones = 0
        for i in range(len(result)):
            if result[i] == 1 and optimal_basis[i] == 1:
                matching_ones += 1
        
        return matching_ones
    
    def inference(self, file):
        """
        Using the file, convert the file into a bipartite graph. Conduct inferencing and make attempted pivots to optimality.
        """
        A, b, c, basis = self.load_lp(file)

        # Calculate the true optimal value
        correct_result = linprog(c, A_ub=A, b_ub=b, method='highs')

        # Check if Solution is feasible
        if not correct_result.success:
            return "The solution is not feasible", 0
        
        # Check if Solution is bounded
        if not correct_result.status == 0:
            return "The solution is unbounded or there was an issue", 0
        
        # Calculate Optimal Basis
        tolerance = 1e-9
        in_basis = [1 if v > tolerance else 0 for v in correct_result.x]
        slack_in_basis = [1 if s >= tolerance else 0 for s in correct_result.slack]
        in_basis = np.array(in_basis)
        slack_in_basis = np.array(slack_in_basis)
        optimal_basis = np.concatenate((in_basis, slack_in_basis))

        # Get the data needed
        variable_features, constraint_features, edge_index, edge_attr = self.bipartite_graph_details(c, A, b)

        # Need to fix this for difference problem size
        no_of_primal=len(c) - len(A)
        variable_basis = basis[:no_of_primal]
        slack_basis = basis[no_of_primal:]
        zeros = np.zeros(len(A), dtype=int)
        cur_basis = np.concatenate((variable_basis, zeros, slack_basis))

        # Initialize a counter for pivots
        num_pivots = 0
        repeat_counter = {}

        start_time = time.time()
        for steps in range(300):
            data = self.get_bipartite(variable_features, constraint_features, edge_index, edge_attr, cur_basis)
            data = data.to(self.device)
            out, optimal = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            feature_0 = out[:, 0]
            copy_0 = feature_0.detach().clone()
            feature_1 = out[:, 1]
            copy_1 = feature_1.detach().clone()

            # Select the entering variable
            for j in range(3000):
                entering = torch.argmax(copy_0-feature_1).item()
                
                if entering >= len(A) and entering <= len(c):
                    copy_0[entering] = float('-inf')
                    continue
                
                if cur_basis[entering] == 1:
                    copy_0[entering] = float('-inf')
                    continue
                else:
                    if steps == 0: 
                        entry_diff = feature_0[entering]-feature_1[entering]
                    elif (feature_0[entering]-feature_1[entering])/entry_diff < 0.4:
                        j = 3000
                    
                    break

            if j == 3000:
                print("No suitable entering variables")
                break
            
            # Select the leaving variable
            for j in range(3000):
                leaving = torch.argmax(copy_1-feature_0).item()
                
                if leaving >= len(A) and leaving <= len(c):
                    copy_1[leaving] = float('-inf')
                    continue
                
                if cur_basis[leaving] == 0:
                    copy_1[leaving] = float('-inf')
                    continue
                else:
                    if steps == 0: 
                        leave_diff = feature_1[leaving]-feature_0[leaving]
                    elif (feature_1[leaving]-feature_0[leaving])/leave_diff < 0.4:
                        j = 3000
                    
                    #print(out[leaving])
                    break
            
            if j == 3000:
                print("No suitable leaving variables")
                break

            current_step = (entering, leaving)

            if current_step in repeat_counter:
                repeat_counter[current_step] += 1
            else:
                repeat_counter[current_step] = 1
            
            if repeat_counter[current_step] > 4:
                break
            elif entering == leaving:
                break
            
            cur_basis[leaving] = 0
            cur_basis[entering] = 1
            num_pivots += 1

        end_time = time.time()
        inference_time = end_time - start_time
        cur_basis = np.array(cur_basis)
        part1 = optimal_basis[:no_of_primal]
        part2 = optimal_basis[no_of_primal+len(A):no_of_primal+len(A)+len(A)]
        optimal_basis = np.concatenate((part1,part2))

        part1 = cur_basis[:no_of_primal]
        part2 = cur_basis[no_of_primal+len(A):no_of_primal+len(A)+len(A)]
        cur_basis = np.concatenate((part1,part2))

        accuracy = self.get_matches(cur_basis, optimal_basis) / len(A)

        predicted_basis_indices = np.where(cur_basis == 1)[0].tolist()
        optimal_basis_indices = np.where(optimal_basis == 1)[0].tolist()

        return predicted_basis_indices, optimal_basis_indices, accuracy*100, inference_time
    
#temp = LTP('/home/ac/website/pivot_learner.pth')
#A, b, c, basis = temp.load_lp('/home/ac/website/example_data.npz')
#print(temp.inference('/home/ac/website/example_data.npz'))