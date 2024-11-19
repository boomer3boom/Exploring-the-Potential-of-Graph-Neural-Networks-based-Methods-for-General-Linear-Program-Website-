Title: Exploring the Potential of Foundational Models on GNN for Linear Programs - Website
This repository contains the code base for the website discussed in paper ____. The website primarily demonstrates how different GNN models generate an initial greedy starting point, from which the Simplex or Interior Point Method can pivot to optimality faster. The website contains three GNNs for users to experiment with: Smart Initial Basis, Learn To Pitovt, and Interior Point Message Passing Neural Network (IPMGNN).

We use Python 3.10.13, and our dependencies are listed in requirements.txt. However, please note that depending on your GPU/CPU, you may need to download different versions of Torch and other Torch-related dependencies. For the GPU, we use NVIDIA GeForce RTX 3090 (2 GPUs available on the system) with Cuda 12.0. 

How to use: 
Once cloned and dependencies are set up, just run app.py (ie python app.py). This will bring you to the local host on port 5000 and you can play with the website. Additionally, you will also have to change the location of files to match your own file locations.

Structure of Code Base:
