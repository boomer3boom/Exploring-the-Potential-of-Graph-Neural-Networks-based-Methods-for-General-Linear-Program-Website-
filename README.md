# Exploring the Potential of Foundational Models on GNN for Linear Programs

Welcome to the repository accompanying the paper **[Insert Paper Title Here]**. This project features a website that demonstrates how various Graph Neural Network (GNN) models can generate an initial greedy starting point, enabling faster convergence to optimality using optimization techniques such as the Simplex Method or Interior Point Method.

The website allows users to experiment with the following pre-trained GNN models:

- **Smart Initial Basis (SIB)**
- **Learn To Pivot (LTP)**
- **Interior Point Message Passing Neural Network (IPMGNN)**

---

## We Tested This Website On:

- **Python version**: 3.10.13  
- **GPU**: NVIDIA GeForce RTX 3090 (2 GPUs available)  
- **CUDA version**: 12.0  

Dependencies are listed in `requirements.txt`. Please ensure you download compatible versions of PyTorch and other Torch-related packages based on your hardware setup.

---

## Getting Started

1. Clone this repository:
   ```bash
   git clone [repository_url]
   cd [repository_folder]

Once cloned and dependencies are set up, just run app.py (ie python app.py). This will bring you to the local host on port 5000 and you can play with the website. Additionally, you will also have to change the location of files to match your own file locations.

Structure of Code Base:
ipmgnn.pt, pivot_learner.pth, primal.pth, and slack.pth are all models corresponding to their respective GNN. These have all been previous trained already, refer to ___. 

IPMGNN.py, LTP.py, SIB.py, arch.py, transformer.py, and IPMGNN_folder are python support files that are required to unpack GNN models so they can digest data into and output results.

app.py, static, and templates are related to the structure of the websites. 
