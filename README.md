# Exploring the Potential of Graph Neural Networks-based Methods for General Linear Program

Welcome to the repository accompanying the paper **Exploring the Potential of Graph Neural Networks-based Methods for General Linear Program**, https://dl.acm.org/doi/10.1145/3701716.3715174 \ This project features a website that demonstrates how various Graph Neural Network (GNN) models can generate an initial greedy starting point, enabling faster convergence to optimality using optimization techniques such as the Simplex Method or Interior Point Method.

The website allows users to experiment with the following pre-trained GNN models:

- **Smart Initial Basis (SIB)**
- **Learn To Pivot (LTP)**
- **Interior Point Message-Passing Graph Neural Network (IPMGNN)**

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
   git clone https://github.com/boomer3boom/Exploring-the-Potential-of-Graph-Neural-Networks-based-Methods-for-General-Linear-Program-Website-
   cd Exploring-the-Potential-of-Graph-Neural-Networks-based-Methods-for-General-Linear-Program-Website-

2. Download the required dependencies: \
   `see requirements.txt on our dependencies but download the packages that are compatible with your setup`

3. You may need to change some file paths based on your directory.

4. Run the app:
   ```bash
   python app.py

---

## Codebase Structure

The codebase is organized into the following components:

### Pre-trained Models
- **`ipmgnn.pt`**, **`pivot_learner.pth`**, **`primal.pth`**, and **`slack.pth`**  
  These files contain pre-trained models for their respective GNN architectures. Refer to [documentation/notes] for details on their training process and usage.

### Python Support Files
- **`IPMGNN.py`**, **`LTP.py`**, **`SIB.py`**, **`arch.py`**, **`transformer.py`**, and the **`IPMGNN_folder`**  
  These files provide the essential Python modules required to load, process, and utilize the GNN models. They handle data ingestion and model inference.

### Web Application Components
- **`app.py`**, **`static`**, and **`templates`**  
  These files and directories define the web application's structure, including backend logic, static assets (e.g., CSS, JavaScript), and HTML templates for rendering the user interface.

