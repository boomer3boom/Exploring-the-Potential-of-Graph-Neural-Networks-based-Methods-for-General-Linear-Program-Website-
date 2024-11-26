import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

from scipy.optimize import linprog
from ML.transformer import *
import numpy as np
import torch
import torch.nn as nn
from ML.arch import *
from numpy.linalg import inv
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time

class SIB():
    def __init__(self, primal_path, slack_path):
        """
        Initilise the primal model that inferences which primal variable should be in basis.
        Initilise the slack model that inferences which slack variable should be in basis.
        """
        # Generate The Model
        input_dim = 2
        hidden_dim = 64
        self.primal_ml = GCN(input_dim, hidden_dim)
        self.slack_ml = GCN(input_dim, hidden_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the Model Parameter
        primal_load = torch.load(primal_path)
        slack_load = torch.load(slack_path)

        self.primal_ml.load_state_dict(primal_load['model_state_dict'])
        self.slack_ml.load_state_dict(slack_load['model_state_dict'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.primal_ml.to(self.device)
        self.slack_ml.to(self.device)

        self.primal_ml.eval()
        self.slack_ml.eval()
    
    def load_lp(self, data):
        """
        Load the file
        """
        print(data)
        try:
            c = data['c']
            A = data['A']
            b = data['b']
        except Exception as e:
            print(e)

        return A, b, c
    
    def conditional_normalize(self, array):
        """
        Normalise the array to be between 0 and 1.
        """
        max_abs_val = np.max(np.abs(array))
        if max_abs_val == 0:
            return array  # Avoid division by zero if the array is all zeros

        normalized_array = array / max_abs_val

        return normalized_array
    
    def normalise(self, A, b, c):
        """
        Normalise the A, b, c matrix.
        """
        A = self.conditional_normalize(A)
        b = self.conditional_normalize(b)
        c = self.conditional_normalize(c)

        return A, b, c
    
    def get_matches(self, result, optimal_basis):
        matching_ones = 0
        for i in range(len(result)):
            if result[i] == 1 and optimal_basis[i] == 1:
                matching_ones += 1
        
        return matching_ones
    
    def sample_vertex(self, A, b):
        """
        Sample the problem's polyhedron and collect sample data points for reference and output the image.
        """
        num_constraints = A.shape[0]
        vertices = []

        if num_constraints < 35:
            # Collect vertices of the feasible region
            for i in range(num_constraints):
                for j in range(i+1, num_constraints):
                    # Solve for the intersection of the ith and jth constraints
                    A_eq = np.array([A[i], A[j]])
                    b_eq = np.array([b[i], b[j]])
                    
                    # Try to find the intersection point
                    try:
                        res = linprog(c=np.zeros(A.shape[1]), A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)]*A.shape[1])
                        if res.success:
                            vertices.append(res.x)
                    except:
                        pass
        else:
            num_samples = 1000

            sampled_pairs = []
            while len(sampled_pairs) < num_samples:
                i, j = np.random.choice(num_constraints, 2, replace=False)
                if (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
                    sampled_pairs.append((i, j))

            # Solve for the intersection points of the sampled pairs
            for i, j in sampled_pairs:
                A_eq = np.array([A[i], A[j]])
                b_eq = np.array([b[i], b[j]])
                
                # Try to find the intersection point
                try:
                    res = linprog(c=np.zeros(A.shape[1]), A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)]*A.shape[1])
                    if res.success:
                        vertices.append(res.x)
                except:
                    pass
        
        return np.array(vertices)
    
    def get_visual(self, data, basis_indices, optimal_indices):
        """
        Output the visualisation of the model
        """
        # Unlock the file
        A, b, c = self.load_lp(data)

        # Normalise to -1 and 1
        A, b, c = self.normalise(A, b, c)
        
        vertices = self.sample_vertex(A, b)
        # Extract the basis matrix B from A
        B_init = A[:, basis_indices]
        B_optimal = A[:, optimal_indices]
        
        # Calculate the inverse of B
        B_init_inv = inv(B_init)
        B_optimal_inv = inv(B_optimal)

        # Calculate the basic variables x^b
        x_b_init = np.dot(B_init_inv, b)
        x_b_optimal = np.dot(B_optimal_inv, b)

        # Initialize x_b array with zeros
        x_b_init_full = np.zeros(vertices.shape[1])
        x_b_optimal_full = np.zeros(vertices.shape[1])

        # Place x_b values into the correct positions
        x_b_init_full[basis_indices] = x_b_init
        x_b_optimal_full[optimal_indices] = x_b_optimal
        
        # Combine vertices with x_b
        combined_vertices = np.vstack([vertices, x_b_init_full, x_b_optimal_full])

        # Apply PCA to reduce dimensions to 3D for visualization
        pca = PCA(n_components=3)
        reduced_combined = pca.fit_transform(combined_vertices)

        # Separate top vertices and x_b in PCA space
        reduced_vertices = reduced_combined[:vertices.shape[0]]
        x_b_init_pca = reduced_combined[vertices.shape[0]:vertices.shape[0] + 1]
        x_b_optimal_pca = reduced_combined[vertices.shape[0] + 1:]

        # Calculate the Euclidean distance from the origin, initial basis, and optimal basis
        origin_distance = np.linalg.norm(reduced_vertices, axis=1)
        
        distance_threshold = max(np.linalg.norm(x_b_optimal_pca), np.linalg.norm(x_b_init_pca)) * 1.25

        # Filter vertices within the distance threshold from key points
        within_threshold = (origin_distance <= distance_threshold)
        filtered_indices = np.where(within_threshold)[0]

        filtered_vertices = reduced_vertices[within_threshold]
        distances = np.linalg.norm(filtered_vertices, axis=1)

        # Calculate the Euclidean distance from the origin for filtered vertices
        if filtered_vertices.shape[0] > 20:
            top_filtered_indices = filtered_indices[np.argsort(distances)[-20:]]
        else:
            top_filtered_indices = filtered_indices[np.argsort(distances)]  # If fewer than 500 vertices, take all
        
        # Number of additional vertices needed to reach 100
        additional_vertices_needed = max(0, 100 - top_filtered_indices.shape[0])

        # Set of indices that have not been selected as top 20
        remaining_indices = np.setdiff1d(filtered_indices, top_filtered_indices)

        # Randomly sample the required number of additional vertices from the remaining vertices
        if remaining_indices.shape[0] > 0:
            random_sample_indices = np.random.choice(
                remaining_indices, size=min(additional_vertices_needed, remaining_indices.shape[0]), replace=False
            )
        else:
            random_sample_indices = np.array([])

        top_filtered_indices = np.array(top_filtered_indices)
        random_sample_indices = np.array(random_sample_indices)
        # Combine top 200 and randomly sampled vertices
        final_indices = np.concatenate((top_filtered_indices, random_sample_indices))
        final_indices = final_indices.astype(int)
        final_vertices = reduced_vertices[final_indices]
        
        return self.display_image(final_vertices, x_b_init_pca, x_b_optimal_pca)
    
    def display_image(self, top_vertices, basis_init, optimal_sol):
        """
        Display the image on the website.
        """
        # Check if there are enough vertices to apply PCA
        if top_vertices.shape[0] > 0:
            # Add the origin to the vertices
            vertices_with_origin = np.vstack([top_vertices])

            # Plot the top 20 prominent vertices in 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            #Plot all vertices with some transparency
            ax.scatter(vertices_with_origin[:, 0], vertices_with_origin[:, 1], vertices_with_origin[:, 2], 
                    c='grey', marker='o', s=30, alpha=0.5)  # Adjusted transparency with alpha

            # Plot the origin point as a more prominent black dot
            ax.scatter(0, 0, 0, c='black', marker='o', s=100, edgecolor=None)  # Larger size, no edge color

            # Plot the basis and optimal solution points
            ax.scatter(basis_init[0][0], basis_init[0][1], basis_init[0][2], c='green', marker='o', s=100, edgecolor=None)
            ax.scatter(optimal_sol[0][0], optimal_sol[0][1], optimal_sol[0][2], c='red', marker='o', s=100, alpha=1, edgecolor=None)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.title('Graph for your LP problem')

            all_vertices = np.vstack([top_vertices, basis_init[0], optimal_sol[0]])

            max_range = np.max(np.abs(all_vertices))
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            plt.tight_layout()  # Adjust layout to fit all elements well

            # Save the plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

            return image_base64
        else:
            return "Could Not Display Image"
    
    def inference(self, data):
        """
        Inference the problem and output the initial basis.
        """
        A, b, c = self.load_lp(data)

        # Check the LP arrays are numpy
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(c, np.ndarray):
            return "Please Include Use Numpy Array", 0

        # Check the LP array contains only numerical numbers
        if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(b.dtype, np.number) or not np.issubdtype(c.dtype, np.number):
            return "Please ensure LP are numerical", 0

        number_of_constraint = A.shape[0]
        number_of_variables = c.shape[0] - number_of_constraint

        # Check the file shape is eligible
        if number_of_constraint > 700 or number_of_variables > 500:
            return "Shape is not Eligible", 0

        # Check if basis exist already
        if not np.all(c[number_of_variables:] == 0):
            return "Please Include Slack Variables", 0

        # Normalise to -1 and 1
        A, b, c = self.normalise(A, b, c)

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
        
        # Pad LP to required shape
        target_c_size = 1200
        target_A_shape = (700, 1200)
        target_b_size = 700

        # Pad c, A, b
        c_slack = np.pad(c, (0, target_c_size - len(c)), 'constant')
        A_slack = np.pad(A, ((0, target_A_shape[0] - A.shape[0]), (0, target_A_shape[1] - A.shape[1])), 'constant')
        b_combined = np.pad(b, (0, target_b_size - len(b)), 'constant')

        # Transform to Bipartite Graph
        input_graph = lp_to_bipartite_graph(c_slack, A_slack, b_combined, 1200, 700)
        input_graph = input_graph.to(self.device)
        start_time = time.time()
        
        # Inference and get probabilities
        with torch.no_grad():
            # Forward pass through the primal model
            primal_output = self.primal_ml(input_graph)

            # Forward pass through the slack model
            slack_output = self.slack_ml(input_graph)

            collected_results = []
            index1 = 0
            index2 = number_of_variables + number_of_constraint
            collected_p = primal_output[index1:index1+number_of_variables]
            collected_results.extend(collected_p.cpu().numpy())
            collected_s = slack_output[index2:index2 + number_of_constraint]
            collected_results.extend(collected_s.cpu().numpy())
        
        end_time = time.time()
        inference_time = end_time - start_time

        top_k_indices = sorted(range(len(collected_results)), key=lambda i: collected_results[i], reverse=True)[:number_of_constraint]

        # Initialize a result list with zeros
        result = [0] * len(collected_results)

        # Set top k probabilities to 1
        for idx in top_k_indices:
            result[idx] = 1

        # Set to numpy array
        result = np.array(result)
        part1 = optimal_basis[:number_of_variables]
        part2 = optimal_basis[number_of_variables+number_of_constraint:number_of_variables+number_of_constraint+number_of_constraint]
        optimal_basis = np.concatenate((part1,part2))

        accuracy = self.get_matches(result, optimal_basis) / number_of_constraint

        predicted_basis_indices = np.where(result == 1)[0].tolist()
        optimal_basis_indices = np.where(optimal_basis == 1)[0].tolist()

        return predicted_basis_indices, optimal_basis_indices, accuracy*100, inference_time