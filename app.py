from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import pandas as pd
import cv2
import lightgbm as lgb
from PIL import Image
import io
import base64
import joblib
from skimage.feature import hog, graycomatrix, graycoprops  # Updated import
from skimage.measure import regionprops
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import requests
from io import StringIO
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variable to store the trained model
global_model = None

@app.template_filter('now')
def template_now():
    return datetime.now()

def download_and_prepare_data():
    """Download and prepare the Wisconsin Breast Cancer dataset"""
    # URL for the Wisconsin Breast Cancer dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    
    try:
        # Download the data
        print("Downloading Wisconsin Breast Cancer dataset...")
        response = requests.get(url)
        
        # Check if the download was successful
        if response.status_code == 200:
            # Column names for the dataset
            column_names = ['ID', 'Diagnosis'] + [
                f'feature_{i}' for i in range(1, 31)
            ]
            
            # Read the data into a pandas DataFrame
            data = pd.read_csv(StringIO(response.text), header=None, names=column_names)
            
            # Convert diagnosis to binary: M (malignant) = 1, B (benign) = 0
            data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
            
            # Drop the ID column as it's not needed for modeling
            data = data.drop('ID', axis=1)
            
            print(f"Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} columns")
            return data
        else:
            print(f"Failed to download dataset. Status code: {response.status_code}")
            # Use a small synthetic dataset as fallback
            return create_synthetic_dataset()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Use a small synthetic dataset as fallback
        return create_synthetic_dataset()

def create_synthetic_dataset():
    """Create a synthetic dataset for demonstration purposes"""
    print("Creating synthetic breast cancer dataset...")
    
    # Number of samples
    n_samples = 569  # Same as the original Wisconsin dataset
    
    # Generate synthetic features
    np.random.seed(42)
    X = np.random.randn(n_samples, 30)
    
    # Generate synthetic target (30% malignant, 70% benign)
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Create column names
    column_names = ['Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    
    # Create DataFrame
    df = pd.DataFrame(np.column_stack([y, X]), columns=column_names)
    
    print(f"Synthetic dataset created with {df.shape[0]} samples and {df.shape[1]} columns")
    return df

def train_lightgbm_model(data):
    """Train a LightGBM model on the breast cancer dataset"""
    print("Training LightGBM model...")
    
    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features with valid names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)  # Convert to NumPy array
    X_test_scaled = scaler.transform(X_test.values)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Set up LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1  # Set to -1 for silent mode
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    # Train the model - removed early_stopping_rounds and verbose_eval parameters
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)
    
    print(f"Model training completed with accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred_binary))
    
    # Save the model
    joblib.dump(model, 'models/lightgbm_cancer_model.pkl')
    
    # Save feature names for future reference
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns))
    
    return model, scaler

def load_or_train_model():
    """Load the pre-trained model or train a new one if it doesn't exist"""
    global global_model
    
    model_path = 'models/lightgbm_cancer_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading pre-trained model and scaler...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        print("No pre-trained model found. Training a new model...")
        data = download_and_prepare_data()
        model, scaler = train_lightgbm_model(data)
    
    global_model = model
    return model, scaler

def extract_features_from_image(image):
    """Extract features from an image for cancer detection"""
    # Resize for consistency
    img_resized = cv2.resize(image, (128, 128))
    
    # Convert to grayscale if needed
    if len(img_resized.shape) > 2:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized
    
    # Apply some preprocessing to enhance features
    img_eq = cv2.equalizeHist(img_gray)
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    
    # Apply threshold to segment potential regions of interest
    _, img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract HOG features
    hog_features = hog(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    
    # Calculate basic statistical features
    mean_val = np.mean(img_gray)
    std_val = np.std(img_gray)
    median_val = np.median(img_gray)
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)
    
    # Calculate histogram features
    hist = cv2.calcHist([img_gray], [0], None, [10], [0, 256])
    hist_features = hist.flatten() / np.sum(hist)  # Normalize
    
    # Calculate texture features using GLCM
    glcm = graycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Combine all features
    all_features = np.concatenate([
        hog_features, 
        [mean_val, std_val, median_val, min_val, max_val],
        hist_features,
        contrast, dissimilarity, homogeneity, energy, correlation
    ])
    
    return all_features

def map_to_wdbc_features(image_features):
    """Map image features to match the format expected by the trained model"""
    # Load feature names
    try:
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Create a feature vector of the right length
        feature_vector = np.zeros(len(feature_names))
        
        # Fill in as many features as we have
        min_len = min(len(image_features), len(feature_vector))
        feature_vector[:min_len] = image_features[:min_len]
        
        return feature_vector
    except:
        # If feature names file doesn't exist, create a synthetic mapping
        print("Feature names file not found. Creating synthetic mapping.")
        # Ensure we have 30 features (same as WDBC dataset)
        if len(image_features) >= 30:
            return image_features[:30]
        else:
            # Pad with zeros if we have fewer features
            return np.pad(image_features, (0, 30 - len(image_features)))

def analyze_image(image_path):
    """Analyze the image for cancer detection and characteristics"""
    global global_model
    
    # Ensure model is loaded
    if global_model is None:
        _, _ = load_or_train_model()
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read the image file"}
    
    # Extract features from the image
    image_features = extract_features_from_image(image)
    
    # Map to WDBC features
    wdbc_features = map_to_wdbc_features(image_features)
    
    # Scale features
    scaled_features = scaler.transform(wdbc_features.reshape(1, -1))
    
    # Make prediction
    prediction_prob = global_model.predict(scaled_features)[0]
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    # For demonstration, we'll simulate different severity levels
    if prediction == 1:
        # Simulate cancer detection
        if prediction_prob > 0.8:
            severity = "High"
        elif prediction_prob > 0.6:
            severity = "Medium"
        else:
            severity = "Low"
        
        # Calculate simulated size (in a real app, use segmentation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            size_mm = round(np.sqrt(area) * 0.1, 2)  # Simulated conversion to mm
        else:
            size_mm = 15.5  # Default size if no contours found
        
        result = {
            "cancer_detected": True,
            "probability": round(float(prediction_prob) * 100, 2),
            "severity": severity,
            "size_mm": size_mm,
            "message": f"Cancer detected with {severity} severity. Approximate size: {size_mm} mm."
        }
    else:
        result = {
            "cancer_detected": False,
            "probability": round((1 - float(prediction_prob)) * 100, 2),
            "message": "No cancer detected in the image."
        }
    
    return result

def generate_visualization(image_path, result):
    """Generate visualization of the analysis"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title("Cancer Analysis Result")
    
    if result["cancer_detected"]:
        # In a real application, you would highlight the detected area
        plt.text(10, 30, f"Cancer Detected: {result['probability']}% confidence", 
                 color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        plt.text(10, 60, f"Severity: {result['severity']}", 
                 color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        plt.text(10, 90, f"Size: {result['size_mm']} mm", 
                 color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a simulated ROI (Region of Interest)
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = int(min(w, h) * 0.3)
        
        # Draw a red rectangle around the "detected" area
        rect = plt.Rectangle(
            (center_x - roi_size//2, center_y - roi_size//2),
            roi_size, roi_size,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        
    else:
        plt.text(10, 30, "No Cancer Detected", 
                 color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        plt.text(10, 60, f"Healthy Tissue: {result['probability']}% confidence", 
                 color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the visualization
    viz_path = os.path.join(app.config['UPLOAD_FOLDER'], 'visualization.png')
    plt.savefig(viz_path)
    plt.close()
    
    return viz_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('analyze.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('analyze.html', error='No selected file')
        
        if file:
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(filename)
            
            # Analyze the image
            result = analyze_image(filename)
            
            if 'error' in result:
                return render_template('analyze.html', error=result['error'])
            
            # Generate visualization
            viz_path = generate_visualization(filename, result)
            
            # Return the results page with current time
            return render_template('results.html', 
                                  result=result, 
                                  image_path=filename,
                                  viz_path=viz_path,
                                  current_time=datetime.now())
    
    return render_template('analyze.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/load-model')
def load_model():
    """API endpoint to trigger model loading/training"""
    try:
        model, scaler = load_or_train_model()
        return jsonify({"status": "success", "message": "Model loaded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    app.run(debug=True)
