import os
from ml_collections import ConfigDict
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from IPMGNN_folder.data.utils import args_set_bool, collate_fn_ip
from IPMGNN_folder.models.hetero_gnn import TripartiteHeteroGNN

import gzip
import pickle
from IPMGNN_folder.solver.linprog import linprog
from torch_scatter import scatter

from scipy.optimize import linprog
from transformer import *
import numpy as np
import torch
import torch.nn as nn
from arch import *
from numpy.linalg import inv
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from IPMGNN_folder.data.dataset import LPDataset
from torch_geometric.transforms import Compose
from IPMGNN_folder.data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
import shutil
import time

class IPMGNN:

    def __init__(self, path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TripartiteHeteroGNN(conv='gcnconv',
                            in_shape=2,
                            pe_dim=0,
                            hid_dim=180,
                            num_conv_layers=8,
                            num_pred_layers=3,
                            num_mlp_layers=4,
                            dropout=0,
                            share_conv_weight=False,
                            share_lin_weight=False,
                            use_norm=True,
                            use_res=False,
                            conv_sequence='cov').to(self.device)
        
        self.model.load_state_dict(torch.load(path))

        self.model.eval()
    
    def conditional_normalize(self, array):
        max_abs_val = np.max(np.abs(array))
        if max_abs_val == 0:
            return array  # Avoid division by zero if the array is all zeros

        normalized_array = array / max_abs_val

        return normalized_array
    
    def normalise(self, A, b, c):
        A = self.conditional_normalize(A)
        b = self.conditional_normalize(b)
        c = self.conditional_normalize(c)

        return A, b, c

    def sample_vertex(self, A, b):
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
            num_samples = 200

            sampled_pairs = []
            while len(sampled_pairs) < num_samples:
                i, j = np.random.choice(num_constraints, 2, replace=False)
                if (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
                    sampled_pairs.append((i, j))

            # Solve for the intersection points of the sampled pairs
            for i, j in sampled_pairs:
                A_ub = np.array([A[i], A[j]])
                b_ub = np.array([b[i], b[j]])
                bounds = (0., 1.)
                # Try to find the intersection point
                try:
                    A_eq = None
                    b_eq = None
                    res = linprog(c=np.zeros(A.shape[1]), 
                            A_ub=A_ub,
                            b_ub=b_ub,
                            A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
                    if res.success:
                        vertices.append(res.x)
                except:
                    pass
        
        return np.array(vertices)
    
    def get_visual(self, A, b, c, pred, opt):


        # Normalise to -1 and 1
        #A, b, c = self.normalise(A, b, c)

        
        vertices = self.sample_vertex(A, b)

        
        # Combine vertices with x_b
        try:
            pred = pred.T
            opt = opt.T
            # Combine vertices with pred and opt
            combined_vertices = np.vstack([vertices, pred, opt])
        except Exception as e:
            print(f"An error occurred: {e}")

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
        # Check if there are enough vertices to apply PCA
        if top_vertices.shape[0] > 0:
            # Add the origin to the vertices
            vertices_with_origin = np.vstack([top_vertices])
            # Compute convex hull including the origin
            #hull = ConvexHull(vertices_with_origin)

            # Plot the top 20 prominent vertices in 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            #Plot all vertices with some transparency
            ax.scatter(vertices_with_origin[:, 0], vertices_with_origin[:, 1], vertices_with_origin[:, 2], 
                    c='grey', marker='o', s=30, alpha=0.5)  # Adjusted transparency with alpha

            # Plot the simplices (triangular faces of the polyhedron)
            #for simplex in hull.simplices:
            #    triangle = vertices_with_origin[simplex]
            #    poly = Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.2)
            #    ax.add_collection3d(poly)

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
    
    def load_lp(self, file):
        ip_pkgs = pickle.load(file)

        return ip_pkgs[0]
    
    def inference(self, A, b, c):
        root = '/home/ac/website'
        ips = []
        pkg_idx = 0
        success_cnt = 0
        bounds = (0., 1.)
        try:
            A_eq = None
            b_eq = None
            A_ub = A
            b_ub = b
            res = linprog(c, 
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
        except:
            return "error", 0, 0, 0
        else:
            if res.success and not np.isnan(res.fun):
                ips.append((A, b, c))
                success_cnt += 1
                with gzip.open(f'{root}/IPMGNN_folder/raw/instance_{pkg_idx}.pkl.gz', "wb") as file:
                    pickle.dump(ips, file)
            else:
                return "error", 0, 0, 0
            
        dataset = LPDataset("/home/ac/website/IPMGNN_folder",
                extra_path=f'{1}restarts_'
                                    f'{0}lap_'
                                    f'{8}steps'
                                    f'{"_upper_" + str(1.0)}',
                upper_bound=1.0,
                rand_starts=1,
                pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=0),
                                                SubSample(8)]))

        # feed problem to model
        test_loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=collate_fn_ip)
        
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                data.to(self.device)
                try:
                    pred, _ = self.model(data)
                except Exception as e:
                    print(e)

                end_time = time.time()
                inference_time = end_time - start_time


                # Caluclate Objective Gap
                pred = pred[:, -8:]
                pred = torch.relu(pred)
                c_times_x = data.obj_const[:, None] * pred
                obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')

                x_gt = data.gt_primals[:, -8:]
                c_times_xgt = data.obj_const[:, None] * x_gt
                obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
                obj_gap = (obj_pred - obj_gt) / obj_gt
                obj_gap = np.abs(obj_gap.detach().cpu().numpy())
    

                Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
                constraint_gap = Ax - data.rhs[:, None]
                constraint_gap = torch.relu(constraint_gap)
                constraint_gap = np.abs(constraint_gap.detach().cpu().numpy())

        os.remove(f'{root}/IPMGNN_folder/raw/instance_{pkg_idx}.pkl.gz')
        try:
            shutil.rmtree('/home/ac/website/IPMGNN_folder/processed_1restarts_0lap_8steps_upper_1.0')
        except Exception as e:
            print(e)

        return pred.detach().cpu().numpy(), x_gt.detach().cpu().numpy(),\
            np.concatenate(obj_gap, axis=0).mean().item(), constraint_gap.mean().item(), inference_time