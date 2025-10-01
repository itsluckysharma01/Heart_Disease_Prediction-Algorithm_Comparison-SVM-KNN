from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained models
try:
    with open('heart_disease_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    with open('heart_disease_knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    
    # Load scaler
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully!")
    except:
        scaler = None
        print("⚠️ Scaler not found - using models without scaling")
    
    models_loaded = True
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    models_loaded = False
    svm_model = None
    knn_model = None
    scaler = None

# Feature names based on the dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Categorical mappings
CATEGORICAL_MAPPINGS = {
    'sex': {'Male': 1, 'Female': 0},
    'cp': {
        'typical angina': 0,
        'atypical angina': 1,
        'non-anginal': 2,
        'asymptomatic': 3
    },
    'fbs': {'True': 1, 'False': 0},
    'restecg': {
        'normal': 0,
        'having ST-T wave abnormality': 1,
        'lv hypertrophy': 2
    },
    'exang': {'True': 1, 'False': 0},
    'slope': {
        'upsloping': 0,
        'flat': 1,
        'downsloping': 2
    },
    'thal': {
        'normal': 1,
        'fixed defect': 2,
        'reversable defect': 3
    }
}

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
        
        # Convert categorical variables
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Ensure all values are numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Handle any remaining NaN values
        df = df.fillna(0)
        
        return df.values[0]
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    """Home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please check if model files exist.'
        }), 500
    
    try:
        # Get form data
        form_data = [
            float(request.form['age']),
            request.form['sex'],
            request.form['cp'],
            float(request.form['trestbps']),
            float(request.form['chol']),
            request.form['fbs'],
            request.form['restecg'],
            float(request.form['thalch']),
            request.form['exang'],
            float(request.form['oldpeak']),
            request.form['slope'],
            int(request.form['ca']),
            request.form['thal']
        ]
        
        # Preprocess the input
        processed_data = preprocess_input(form_data)
        
        if processed_data is None:
            return jsonify({'error': 'Error preprocessing input data'}), 400
        
        # Reshape for prediction
        input_data = processed_data.reshape(1, -1)
        
        # Scale the data if scaler is available
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Make predictions
        svm_prediction = svm_model.predict(input_data)[0]
        knn_prediction = knn_model.predict(input_data)[0]
        
        # Get prediction probabilities if available
        svm_proba = None
        knn_proba = None
        
        try:
            if hasattr(svm_model, 'predict_proba'):
                svm_proba = svm_model.predict_proba(input_data)[0]
            elif hasattr(svm_model, 'decision_function'):
                # For SVM, convert decision function to probability-like score
                decision = svm_model.decision_function(input_data)[0]
                # Sigmoid function to convert to probability-like score
                svm_confidence = 1 / (1 + np.exp(-decision))
                svm_proba = [1 - svm_confidence, svm_confidence] if svm_prediction == 1 else [svm_confidence, 1 - svm_confidence]
        except:
            pass
            
        try:
            if hasattr(knn_model, 'predict_proba'):
                knn_proba = knn_model.predict_proba(input_data)[0]
        except:
            pass
        
        # Prepare response
        response = {
            'svm_prediction': int(svm_prediction),
            'knn_prediction': int(knn_prediction),
            'svm_result': 'Heart Disease Detected' if svm_prediction > 0 else 'No Heart Disease',
            'knn_result': 'Heart Disease Detected' if knn_prediction > 0 else 'No Heart Disease',
            'agreement': 'Yes' if (svm_prediction > 0) == (knn_prediction > 0) else 'No'
        }
        
        # Add probabilities if available
        if svm_proba is not None:
            response['svm_confidence'] = float(svm_proba[1] if len(svm_proba) > 1 else svm_proba[0])
        
        if knn_proba is not None:
            response['knn_confidence'] = float(knn_proba[1] if len(knn_proba) > 1 else knn_proba[0])
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'svm_available': svm_model is not None,
        'knn_available': knn_model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Heart Disease Prediction App...")
    print(f"📊 Models loaded: {models_loaded}")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)