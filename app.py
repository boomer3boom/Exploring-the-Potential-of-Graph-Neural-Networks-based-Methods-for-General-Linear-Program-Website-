from flask import Flask, request, jsonify, render_template
import numpy as np
from flask_cors import CORS
from SIB import *

app = Flask(__name__)
CORS(app)

primal_parameter = '/Users/allanchuang/Documents/Uni/2024/Winter research/website/primal.pth'
slack_parameter = '/Users/allanchuang/Documents/Uni/2024/Winter research/website/slack.pth'
learner = SIB(primal_path=primal_parameter, slack_path=slack_parameter)

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
    try:
        data = np.load(file)
        Predicted_Basis, optimal_basis, accuracy = learner.inference(data)
        A, b, c = learner.load_lp(data)
        image_base64 = learner.get_visual(data, Predicted_Basis, optimal_basis)
        if isinstance(Predicted_Basis, str):
            result = {
                'message': Predicted_Basis,
            }
        else:
            result = {
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

if __name__ == '__main__':
    app.run(debug=True)