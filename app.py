from flask import Flask, request, jsonify, render_template
import numpy as np
from flask_cors import CORS
from SIB import *
from IPMGNN import *
from LTP import *
from IPMGNN_folder.data.dataset import LPDataset
from torch_geometric.transforms import Compose
from IPMGNN_folder.data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
import time

app = Flask(__name__)
CORS(app)

# SIB
primal_parameter = '/home/ac/website/primal.pth'
slack_parameter = '/home/ac/website/slack.pth'
SIB_learner = SIB(primal_path=primal_parameter, slack_path=slack_parameter)


# LTP
LTP_parameter = '/home/ac/website/pivot_learner.pth'
LTP_learner = LTP(path=LTP_parameter)

# IPMMGNN
IPMGNN_parameter = '/home/ac/website/ipmgnn.pt'
IPMGNN_learner = IPMGNN(path=IPMGNN_parameter)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/process_lp', methods=['POST'])
def process_lp():
    if 'lp-file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['lp-file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    selected_method = request.form.get('method')  # Get the selected method from the dropdown
    
    # method1 is SIB
    if selected_method == 'method1':
        try:
            data = np.load(file)
            Predicted_Basis, optimal_basis, accuracy, inference_time = SIB_learner.inference(data)
            if isinstance(Predicted_Basis, str):
                result = {
                    'message': Predicted_Basis,
                }
            else:
                A, b, c = SIB_learner.load_lp(data)
                image_base64 = SIB_learner.get_visual(data, Predicted_Basis, optimal_basis)
                result = {
                    'inference_time': str(inference_time),
                    'Accuracy': str(accuracy) + "%",
                    'SIB Basis': str(Predicted_Basis),
                    'Optimal Basis': str(optimal_basis),
                    'A': str(A),
                    'b': str(b),
                    'c': str(c),
                    'image_base64': image_base64
                }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # method 2 is LTP
    elif selected_method == 'method2':
        try:
            data = np.load(file)
            Predicted_Basis, optimal_basis, accuracy, inference_time = LTP_learner.inference(data)
            if isinstance(Predicted_Basis, str):
                result = {
                    'message': Predicted_Basis,
                }
            else:
                A, b, c = SIB_learner.load_lp(data)
                image_base64 = SIB_learner.get_visual(data, Predicted_Basis, optimal_basis)
                result = {
                    'inference_time': str(inference_time),
                    'Accuracy': str(accuracy) + "%",
                    'SIB Basis': str(Predicted_Basis),
                    'Optimal Basis': str(optimal_basis),
                    'A': str(A),
                    'b': str(b),
                    'c': str(c),
                    'image_base64': image_base64
                }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    # method 3 is IPMGNN
    elif selected_method == 'method3':
        try:
            #data = np.load(file)
            A, b, c = IPMGNN_learner.load_lp(file)
            pred, optimal, obj_gap, con_gap, inference_time = IPMGNN_learner.inference(A, b, c)
            image_base64 = IPMGNN_learner.get_visual(A, b, c, pred, optimal)

            pred = pred.tolist()
            optimal = optimal.tolist()

            if isinstance(pred, str):
                result = {
                    'message': Predicted_Basis,
                }
            else:
                result = {
                        'inference_time': str(inference_time),
                        'obj_gap': str(obj_gap*100) + "%",
                        'con_gap': str(con_gap),
                        'pred': str(pred),
                        'optimal': str(optimal),
                        'A': str(A),
                        'b': str(b),
                        'c': str(c),
                        'image_base64': image_base64
                    }
                
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)